clear
clc

% load pressure variable in firth of clyde

load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_mslp_foc_2020.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_mslp_foc_2021.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_mslp_foc_2022.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_mslp_foc_2023.mat

% load u10 variable in firth of clyde

load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_u10_foc_2020.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_u10_foc_2021.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_u10_foc_2022.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_u10_foc_2023.mat

% load v10 variable in firth of clyde

load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_v10_foc_2020.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_v10_foc_2021.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_v10_foc_2022.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_v10_foc_2023.mat

% load vc principal components

load ..\O8_abmc_visual_crossing_forecast_variables\O8_3_visual_crossing_pcs\vc_pcs_2020.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_3_visual_crossing_pcs\vc_pcs_2021.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_3_visual_crossing_pcs\vc_pcs_2022.mat
load ..\O8_abmc_visual_crossing_forecast_variables\O8_3_visual_crossing_pcs\vc_pcs_2023.mat

% load surge timetable

load ..\O4_millport_surge\ntslf_tt_millport_surge.mat
start_index = find(ntslf_tt_millport_surge.Time == datetime(2020,1,1,0,0,0));
end_index = find(ntslf_tt_millport_surge.Time == datetime(2023,12,31,23,0,0));
surge = ntslf_tt_millport_surge.residual(start_index:end_index);

% merge arrays

designmatrix = [[vc_pcs_2020; vc_pcs_2021; vc_pcs_2022; vc_pcs_2023], [vc_mslp_foc_2020;vc_mslp_foc_2021;vc_mslp_foc_2022;vc_mslp_foc_2023], [vc_u10_foc_2020;vc_u10_foc_2021;vc_u10_foc_2022;vc_u10_foc_2023], [vc_v10_foc_2020;vc_v10_foc_2021;vc_v10_foc_2022;vc_v10_foc_2023], surge];

% normalise designmatrix by era5 desingmatrix mean and standard deviation

load ..\O5_designmatrix\dm_mean.mat
load ..\O5_designmatrix\dm_std.mat

normalised_designmatrix = (designmatrix - repmat(dm_mean,[length(designmatrix), 1]))./repmat(dm_std,[length(designmatrix), 1]);

s=size(normalised_designmatrix);

% fill 3D array
dm_abmc = zeros([s(1), s(2), 49]);
for i=25:1:length(normalised_designmatrix)-24
    i
    dm_abmc(i,:,:) = normalised_designmatrix(i-24:i+24,:)';
end

dm_abmc(1:24,:,:) = nan;
dm_abmc(end-24:end,:,:) = nan;
% remove nan
%index = sum(isnan(dm_abmc), [2,3]) == 0;
%dm_abmc = dm_abmc(index, :, :);

dm_abmc_2020 = dm_abmc(1:8784,:,:);
dm_abmc_2021 = dm_abmc(8785:17544,:,:);
dm_abmc_2022 = dm_abmc(17545:26304,:,:);
dm_abmc_2023 = dm_abmc(26305:end,:,:);

% save 
save('dm_abmc_2020.mat', "dm_abmc_2020")
save('dm_abmc_2021.mat', "dm_abmc_2021")
save('dm_abmc_2022.mat', "dm_abmc_2022")
save('dm_abmc_2023.mat', "dm_abmc_2023")
