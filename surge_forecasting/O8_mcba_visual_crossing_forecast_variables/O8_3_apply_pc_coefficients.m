%%%%%%%%%%%%%%%%
%%% FORECAST %%% 
%%%%%%%%%%%%%%%%

load ..\O3_pca\era5_pc_coeffs.mat

load O8_2_visual_crossing_inputs_processed_to_era5_variables\fc_u10_2020.mat
load O8_2_visual_crossing_inputs_processed_to_era5_variables\fc_v10_2020.mat
load O8_2_visual_crossing_inputs_processed_to_era5_variables\fc_mslp_2020.mat

var = [fc_u10_2020, fc_v10_2020, fc_mslp_2020];

meanav = squeeze(mean(var, 1, 'omitnan'));
stdav = squeeze(std(var, 1, 'omitnan'));

pcs = nan(size(var));

for i = 1:length(var)
    try
        % must subtract th mean before applying pc coefficients
        pcs(i,:) =((var(i,:)-meanav)./stdav)*era5_pc_coeffs;
    catch
        continue
    end
end

fc_pcs_2020 = pcs(:,1:18); 

save("O8_3_visual_crossing_pcs\fc_pcs_2020.mat","fc_pcs_2020")

clearvars -except era5_pc_coeffs

%%%%%%%%%%%%%%%
%%% NOWCAST %%% 
%%%%%%%%%%%%%%%

load O8_2_visual_crossing_inputs_processed_to_era5_variables\nc_u10_2020.mat
load O8_2_visual_crossing_inputs_processed_to_era5_variables\nc_v10_2020.mat
load O8_2_visual_crossing_inputs_processed_to_era5_variables\nc_mslp_2020.mat

var = [nc_u10_2020, nc_v10_2020, nc_mslp_2020];

meanav = squeeze(mean(var, 1, 'omitnan'));
stdav = squeeze(std(var, 1, 'omitnan'));

pcs = nan(size(var));

for i = 1:length(var)
    try
        % must subtract th mean before applying pc coefficients
        pcs(i,:) =((var(i,:)-meanav)./stdav)*era5_pc_coeffs;
    catch
        continue
    end
end

nc_pcs_2020 = pcs(:,1:18); 

save("O8_3_visual_crossing_pcs\nc_pcs_2020.mat","nc_pcs_2020")