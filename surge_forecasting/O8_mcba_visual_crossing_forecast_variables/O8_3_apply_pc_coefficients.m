%%% 

load ..\O3_pca\era5_pc_coeffs.mat

load O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_u10_2020.mat
load O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_v10_2020.mat
load O8_2_visual_crossing_inputs_processed_to_era5_variables\vc_mslp_2020.mat

var = [vc_u10_2020, vc_v10_2020, vc_mslp_2020];

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

vc_pcs_2020 = pcs(:,1:18); 

save("O8_3_visual_crossing_pcs\vc_pcs_2020.mat","vc_pcs_2020")