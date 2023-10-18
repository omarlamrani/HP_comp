#include "ExplicitScheme.h"

#include <iostream>
#include <omp.h>

#define POLY2(i, j, imin, jmin, ni) (((i) - (imin)) + (((j)-(jmin)) * (ni)))


ExplicitScheme::ExplicitScheme(const InputFile* input, Mesh* m) :
    mesh(m)
{
    int nx = mesh->getNx()[0];
    int ny = mesh->getNx()[1];
}

void ExplicitScheme::doAdvance(const double dt)
{
    
    double pre_diffusion_run = omp_get_wtime();
    diffuse(dt);            
    double post_diffusion_run = omp_get_wtime();
    printf("diffusion took %g seconds\n", post_diffusion_run - pre_diffusion_run);

    
    double pre_reset_run = omp_get_wtime();
    reset();           
    double post_reset_run = omp_get_wtime();
    printf("reset took %g seconds\n", post_reset_run - pre_reset_run);

    
    double pre_update_bound_run = omp_get_wtime();
    updateBoundaries();           
    double post_update_bound_run = omp_get_wtime();
    printf("boundary_update took %g seconds\n", post_update_bound_run - pre_update_bound_run);
}

void ExplicitScheme::updateBoundaries()
{   
    omp_set_num_threads(4);
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < 4; i++) {
        reflectBoundaries(i);
    } 
}

void ExplicitScheme::init()
{
    updateBoundaries();
}

void ExplicitScheme::reset()
{
    double* u0 = mesh->getU0();
    double* u1 = mesh->getU1();
    int x_min = mesh->getMin()[0];
    int x_max = mesh->getMax()[0];
    int y_min = mesh->getMin()[1]; 
    int y_max = mesh->getMax()[1]; 

    int nx = mesh->getNx()[0]+2;

    #pragma omp parallel for collapse(2) schedule(guided)//collapse the two loops into a single parallel region
    // int num_threads = omp_get_num_threads();
    // std::cout << "number_threads Exp_Scheme reset " << num_threads << std::endl;
    /* more efficient if the amount of work done inside the loop is large 
    relative to the overhead of setting up and synchronizing the parallel threads*/

    // schedule(dynamic, 100); increase granularity of work with chunk size of 100

    /*give each thread a chunk of 100 iterations to work on, which can help to reduce 
    the overhead */

    for(int k = y_min-1; k <= y_max+1; k++) {
        for(int j = x_min-1; j <=  x_max+1; j++) {
            int i = POLY2(j,k,x_min-1,y_min-1,nx);
            u0[i] = u1[i];
        }
    }
}

void ExplicitScheme::diffuse(double dt)
{
    double* u0 = mesh->getU0();
    double* u1 = mesh->getU1();
    int x_min = mesh->getMin()[0];
    int x_max = mesh->getMax()[0];
    int y_min = mesh->getMin()[1]; 
    int y_max = mesh->getMax()[1]; 
    double dx = mesh->getDx()[0];
    double dy = mesh->getDx()[1];

    int nx = mesh->getNx()[0]+2;

    double rx = dt/(dx*dx);
    double ry = dt/(dy*dy);

    double pre_for_diffuse_run = omp_get_wtime();
    

    #pragma omp parallel for schedule(guided)//private(u1)// this only can cause results to be inexact 
    // ordered 
    /* make sure updates to u1 are done in the same order as if it were sequential code*/
    // private(u1) 
    /* can cause race conditions */
    // int num_threads = omp_get_num_threads();
    // std::cout << "number_threads diffuse " << num_threads << std::endl;
    for(int k=y_min; k <= y_max; k++) {
        for(int j=x_min; j <= x_max; j++) {

            int n1 = POLY2(j,k,x_min-1,y_min-1,nx);
            int n2 = POLY2(j-1,k,x_min-1,y_min-1,nx);
            int n3 = POLY2(j+1,k,x_min-1,y_min-1,nx);
            int n4 = POLY2(j,k-1,x_min-1,y_min-1,nx);
            int n5 = POLY2(j,k+1,x_min-1,y_min-1,nx);

            // #pragma omp critical // make sure this part is only ran by one thread at a time
            //  
            u1[n1] = (1.0-2.0*rx-2.0*ry)*u0[n1] + rx*u0[n2] + rx*u0[n3]+ ry*u0[n4] + ry*u0[n5];
        }
    }

    double post_for_diffuse_run = omp_get_wtime();

    printf("full_for_diffuse took %g seconds\n", post_for_diffuse_run - pre_for_diffuse_run);
}

void ExplicitScheme::reflectBoundaries(int boundary_id)
{
    double* u0 = mesh->getU0();
    int x_min = mesh->getMin()[0];
    int x_max = mesh->getMax()[0];
    int y_min = mesh->getMin()[1]; 
    int y_max = mesh->getMax()[1]; 

    int nx = mesh->getNx()[0]+2;

    switch(boundary_id) {
        case 0: 
            {
            /* top */
        
                // double pre_for_top_run = omp_get_wtime();

                #pragma omp parallel for schedule(guided)//private(u0) // can be problematic in terms of overhead, check if race poss 
                // DOES LOWER THE TOTAL TEMPERATURE
                // int num_threads = omp_get_num_threads();
                // std::cout << "number_threads top " << num_threads << std::endl;
                for(int j = x_min; j <= x_max; j++) {
                    int n1 = POLY2(j, y_max, x_min-1, y_min-1, nx);
                    int n2 = POLY2(j, y_max+1, x_min-1, y_min-1, nx);
                /* very dangerous as across various threads n2 and n1 can have similar values;
                THIS can cause race conditions and completly mess up the results*/
                    u0[n2] = u0[n1];
                }
                // double post_for_top_run = omp_get_wtime();

                // printf("full_top_loop took %g seconds\n", post_for_top_run - pre_for_top_run);
            } break;
        case 1:
            /* right */
            {
                // double pre_for_right_run = omp_get_wtime();

                
                #pragma omp parallel for schedule(guided)//private(u0)
                // int num_threads = omp_get_num_threads();
                // std::cout << "number_threads right " << num_threads << std::endl;
                for(int k = y_min; k <= y_max; k++) {
                    int n1 = POLY2(x_max, k, x_min-1, y_min-1, nx);
                    int n2 = POLY2(x_max+1, k, x_min-1, y_min-1, nx);

                    u0[n2] = u0[n1];
                }

                // double post_for_right_run = omp_get_wtime();

                // printf("full_right_loop took %g seconds\n", post_for_right_run - pre_for_right_run);
            } break;
        case 2: 
            /* bottom */
            {
                // double pre_for_bottom_run = omp_get_wtime();

                
                #pragma omp parallel for schedule(guided)//private(u0)
                // int num_threads = omp_get_num_threads();
                // std::cout << "number_threads bottom " << num_threads << std::endl;
                for(int j = x_min; j <= x_max; j++) {
                    int n1 = POLY2(j, y_min, x_min-1, y_min-1, nx);
                    int n2 = POLY2(j, y_min-1, x_min-1, y_min-1, nx);

                    u0[n2] = u0[n1];
                }
                // double post_for_bottom_run = omp_get_wtime();

                // printf("full_bottom_loop took %g seconds\n", post_for_bottom_run - pre_for_bottom_run);
            } break;
        case 3: 
            /* left */
            {
                // double pre_for_left_run = omp_get_wtime();

                #pragma omp parallel for schedule(guided)//private(u0)
                // int num_threads = omp_get_num_threads();
                // std::cout << "number_threads left " << num_threads << std::endl;
                for(int k = y_min; k <= y_max; k++) {
                    int n1 = POLY2(x_min, k, x_min-1, y_min-1, nx);
                    int n2 = POLY2(x_min-1, k, x_min-1, y_min-1, nx);

                    u0[n2] = u0[n1];
                }

                // double post_for_left_run = omp_get_wtime();

                // printf("full_left_loop took %g seconds\n", post_for_left_run - pre_for_left_run);
            } break;
        default: std::cerr << "Error in reflectBoundaries(): unknown boundary id (" << boundary_id << ")" << std::endl;
    }
}
