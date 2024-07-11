load ..\O1_location_selection\domainlocs.mat

% % get filenames for u10
% filenames_u10=[];
% dinfo = dir('u10_wind_at_domain_locs\*.mat');
% for K=1:length(dinfo)
%   iname = string(dinfo(K).name);
%   filenames_u10=[filenames_u10;iname];
% end
% 
% % get filenames for v10
% filenames_v10=[];
% dinfo = dir('v10_wind_at_domain_locs\*.mat');
% for K=1:length(dinfo)
%   iname = string(dinfo(K).name);
%   filenames_v10=[filenames_v10;iname];
% end

% get filenames for mslp
filenames_mslp=[];
dinfo = dir('..\O2_era5_variables\msl_pressure_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_mslp=[filenames_mslp;iname];
end

% load files and concatenate u10
for i=41:length(filenames_u10)
    %u10 = load('u10_wind_at_domain_locs\'+filenames_u10(i),"-mat");
    %v10 = load('v10_wind_at_domain_locs\'+filenames_v10(i),"-mat");
    msl = load('..\O2_era5_variables\msl_pressure_at_domain_locs\'+filenames_mslp(i),"-mat");
    
    %u10 = u10.U10;
    %v10 = v10.V10;
    msl = msl.MSL;
    time = msl.dtime;

    %if (i == 1)
    M = msl.mslord(:,362);
    T = time;
    %else
    %    M = [M; msl.mslord(:,362)];
    %    T = [T; time];
    %end
end

era5_2020_msl_pressure = timetable(M,'RowTimes',T);
era5_2020_msl_pressure = retime(era5_2020_msl_pressure,'hourly');

% save residual timetable
save('era5_2020_msl_pressure.mat', "era5_2020_msl_pressure")

