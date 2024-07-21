% this script will read Firth of CLyde local pressure in Pa

cd O8_2_visual_crossing_inputs_processed_to_era5_variables

req = 362;

load fc_mslp_2020.mat
fc_mslp_foc_2020 = max(fc_mslp_2020, [], 2) - fc_mslp_2020(:, req);
save("fc_mslp_foc_2020.mat","fc_mslp_foc_2020")
clearvars -except req

load nc_mslp_2020.mat
nc_mslp_foc_2020 = max(nc_mslp_2020, [], 2) - nc_mslp_2020(:, req);
save("nc_mslp_foc_2020.mat","nc_mslp_foc_2020")
clearvars -except req

load fc_u10_2020.mat
fc_u10_foc_2020 = fc_u10_2020(:, req);
save("fc_u10_foc_2020.mat","fc_u10_foc_2020")
clearvars -except req

load nc_u10_2020.mat
nc_u10_foc_2020 = nc_u10_2020(:, req);
save("nc_u10_foc_2020.mat","nc_u10_foc_2020")
clearvars -except req

load fc_v10_2020.mat
fc_v10_foc_2020 = fc_v10_2020(:, req);
save("fc_v10_foc_2020.mat","fc_v10_foc_2020")
clearvars -except req

load nc_v10_2020.mat
nc_v10_foc_2020 = nc_v10_2020(:, req);
save("nc_v10_foc_2020.mat","nc_v10_foc_2020")
clearvars -except req