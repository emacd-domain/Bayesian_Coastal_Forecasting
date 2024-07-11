load ..\O1_location_selection\domainlocs.mat

% get filenames for u10
filenames_u10=[];
dinfo = dir('..\O2_era5_variables\u10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_u10=[filenames_u10;iname];
end

% load files and concatenate u10
for i=1:length(filenames_u10)
    u10 = load('..\O2_era5_variables\u10_wind_at_domain_locs\'+filenames_u10(i),"-mat");

    u10 = u10.U10;

    time = u10.dtime;

    if (i == 1)
        M = u10.u10ord(:,362);
        T = time;
    else
        M = [M; u10.u10ord(:,362)];
        T = [T; time];
    end
end

era5_tt_u10 = timetable(M,'RowTimes',T);
era5_tt_u10 = retime(era5_tt_u10,'hourly');

% save residual timetable
save('era5_tt_u10.mat', "era5_tt_u10")

% get filenames for v10
filenames_v10=[];
dinfo = dir('..\O2_era5_variables\v10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_v10=[filenames_v10;iname];
end

% load files and concatenate v10
for i=1:length(filenames_v10)
    v10 = load('..\O2_era5_variables\v10_wind_at_domain_locs\'+filenames_v10(i),"-mat");
    v10 = v10.V10;
    time = v10.dtime;

    if (i == 1)
        M = v10.v10ord(:,362);
        T = time;
    else
        M = [M; v10.v10ord(:,362)];
        T = [T; time];
    end
end

era5_tt_v10 = timetable(M,'RowTimes',T);
era5_tt_v10 = retime(era5_tt_v10,'hourly');

% save residual timetable
save('era5_tt_v10.mat', "era5_tt_v10")

load ..\O1_location_selection\domainlocs.mat

% get filenames for mslp
filenames_mslp=[];
dinfo = dir('..\O2_era5_variables\msl_pressure_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_mslp=[filenames_mslp;iname];
end

% load files and concatenate u10
for i=1:length(filenames_u10)

    msl = load('..\O2_era5_variables\msl_pressure_at_domain_locs\'+filenames_mslp(i),"-mat");
    
    msl = msl.MSL;
    time = msl.dtime;

    if (i == 1)
        M = max(msl.mslord, [], 2) - msl.mslord(:,362);
        T = time;
    else
        M = [M; max(msl.mslord, [], 2) - msl.mslord(:,362)];
        T = [T; time];
    end
end

era5_tt_msl_pressure = timetable(M,'RowTimes',T);
era5_tt_msl_pressure = retime(era5_tt_msl_pressure,'hourly');

% save residual timetable
save('era5_tt_msl_pressure.mat', "era5_tt_msl_pressure")