#include "Driver.h"

#include <omp.h>
#include <iostream>

    Driver::Driver(const InputFile* input, const std::string& pname)
: problem_name(pname)
{

    std::cout << "+++++++++++++++++++++" << std::endl;
    std::cout << "  Running deqn v0.1  " << std::endl;
#ifdef DEBUG
    std::cout << "- input file: " << problem_name << std::endl;
#endif
    // read data from file 
    dt_max = input->getDouble("dt_max",  0.2);
    dt = input->getDouble("initial_dt", 0.2);

    t_start = input->getDouble("start_time", 0.0);
    t_end = input->getDouble("end_time", 10.0);

    vis_frequency = input->getInt("vis_frequency",-1);
    summary_frequency = input->getInt("summary_frequency", 1);

#ifdef DEBUG
    std::cout << "- dt_max: " << dt_max << std::endl;
    std::cout << "- initial_dt: " << dt << std::endl;
    std::cout << "- start_time: " << t_start << std::endl;
    std::cout << "- end_time: " << t_end << std::endl;
    std::cout << "- vis_frequency: " << vis_frequency << std::endl;
    std::cout << "- summary_frequency: " << summary_frequency << std::endl;
#endif
    std::cout << "+++++++++++++++++++++" << std::endl;
    std::cout << std::endl;

    
    double pre_mesh_run = omp_get_wtime();

    mesh = new Mesh(input); 
    double post_mesh_run = omp_get_wtime();

    printf("new_mesh took %g seconds\n", post_mesh_run - pre_mesh_run);

    
    double pre_diffusion_run = omp_get_wtime();

    diffusion = new Diffusion(input, mesh);
    double post_diffusion_run = omp_get_wtime();

    printf("new_diffusion took %g seconds\n", post_diffusion_run - pre_diffusion_run);
    
    // writer = new VtkWriter(pname, mesh);

    // /* Initial mesh dump */
    // if(vis_frequency != -1)
    //     writer->write(0, 0.0);
}

Driver::~Driver() {
    delete mesh;
    delete diffusion;
    // delete writer;
}

void Driver::run() {

    int step = 0;
    double t_current;

    // #pragma omp parallel for
    for(t_current = t_start; t_current < t_end; t_current += dt) {
        step = t_current/dt + 1;

        std::cout << "+ step: " << step << ", dt:   " << dt << std::endl;


        double pre_docycle_run = omp_get_wtime();

        // #pragma omp parallel
        // {
        diffusion->doCycle(dt);
        // }

        // diffusion->doCycle(dt);
        double post_docycle_run = omp_get_wtime();

        printf("do_cycle took %g seconds\n", post_docycle_run - pre_docycle_run);
        

        // if(step % vis_frequency == 0 && vis_frequency != -1)
        //     writer->write(step, t_current);
        if(step % summary_frequency == 0 && summary_frequency != -1) {
            // double pre_temp_run = omp_get_wtime();

            // double temperature = 0.0;
            // #pragma omp parallel reduction(+:temperature)
            // {
            //     temperature = mesh->getTotalTemperature();
            // }

            double pre_temp_run = omp_get_wtime();

            double temperature = mesh->getTotalTemperature();
            
            double post_temp_run = omp_get_wtime();

            printf("total_temp took %g seconds\n", post_temp_run - pre_temp_run);
            std::cout << "+\tcurrent total temperature: " << temperature << std::endl;
        }

    }

    // double post_docycle_run = omp_get_wtime();

    // printf("full_docycle took %g seconds\n", post_docycle_run - pre_docycle_run);

    // if(step % vis_frequency != 0 && vis_frequency != -1)
    //     writer->write(step, t_current);

    std::cout << std::endl;
    std::cout << "+++++++++++++++++++++" << std::endl;
    std::cout << "   Run completete.   " << std::endl;
    std::cout << "+++++++++++++++++++++" << std::endl;
}
