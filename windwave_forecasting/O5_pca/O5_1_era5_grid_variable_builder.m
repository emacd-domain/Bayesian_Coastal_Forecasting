% get filenames for u10
filenames_u10=[];
dinfo = dir('..\O2_era5_variables\u10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_u10=[filenames_u10;iname];
end

% get filenames for v10
filenames_v10=[];
dinfo = dir('..\O2_era5_variables\v10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_v10=[filenames_v10;iname];
end

% load files and concatenate u10
for i=1:3%length(filenames_u10)
    var = load('..\O2_era5_variables\u10_wind_at_domain_locs\'+filenames_u10(i),"-mat");
    if (i == 1)
        U = var.U10;
    else
        U = [U; var.U10];
    end
end

% load files and concatenate v10
for i=1:3%length(filenames_v10)
    var = load('..\O2_era5_variables\v10_wind_at_domain_locs\'+filenames_v10(i),"-mat");
    if (i == 1)
        V = var.V10;
    else
        V = [V; var.V10];
    end
end


% merge files
era5_domain_inputs = synchronize(U,V);
era5_domain_inputs=retime(era5_domain_inputs,'hourly');
era5_domain_inputs=[era5_domain_inputs.u10ord,era5_domain_inputs.v10ord,era5_domain_inputs.mslord];

era5_mean = mean(era5_domain_inputs, 1);
era5_std = std(era5_domain_inputs, 1);

norm_era5_domain_inputs = (era5_domain_inputs - era5_mean) ./ era5_std;

% apply pca
[coeff, score, latent, tsquared, explained] = pca(norm_era5_domain_inputs);
era5_pcs_cumulative_explained=cumsum(explained);
index=find(era5_pcs_cumulative_explained<99.5);
era5_pc_coeffs = coeff;
era5_pcs_995 = score(:,index); 

% save pca arrays

save('era5_mean.mat', 'era5_mean')
save('era5_std.mat', 'era5_std')
save('era5_pc_coeffs.mat', 'era5_pc_coeffs')
%save('era5_pcs_995.mat', 'era5_pcs_995')
save('era5_pcs_cumulative_explained.mat', 'era5_pcs_cumulative_explained')
