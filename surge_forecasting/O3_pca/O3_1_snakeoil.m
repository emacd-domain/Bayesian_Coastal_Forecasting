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

% get filenames for mslp
filenames_mslp=[];
dinfo = dir('..\O2_era5_variables\msl_pressure_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_mslp=[filenames_mslp;iname];
end


% load files and concatenate u10
for i=1:length(filenames_u10)
    i
    var = load('..\O2_era5_variables\u10_wind_at_domain_locs\'+filenames_u10(i),"-mat");
    U = var.U10;
    var = load('..\O2_era5_variables\v10_wind_at_domain_locs\'+filenames_v10(i),"-mat");
    V = var.V10;
    var = load('..\O2_era5_variables\msl_pressure_at_domain_locs\'+filenames_mslp(i),"-mat");
    P = var.MSL;

    era5_domain_inputs = synchronize(U,V,P);
    era5_domain_inputs=retime(era5_domain_inputs,'hourly');
    era5_domain_inputs=[era5_domain_inputs.u10ord,era5_domain_inputs.v10ord,era5_domain_inputs.mslord];

    norm_era5_domain_inputs = (era5_domain_inputs - era5_mean) ./ era5_std;

    pcs = norm_era5_domain_inputs*coeff;

    pcs = pcs(:, 1:18);

    save('pcs_by_year\era5_'+string(1979+i)+'.mat', 'pcs')
end

% get filenames for mslp
filenames=[];
dinfo = dir('pcs_by_year\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end

all_vars = [];
for i=1:length(filenames_u10)
    i
    var = load('pcs_by_year\'+filenames(i),"-mat");
    all_vars = [all_vars;var.pcs];
end

era5_pcs_1980_2019 = all_vars(1:350664,:);
era5_pcs_2020 = all_vars(350665:end,:);
