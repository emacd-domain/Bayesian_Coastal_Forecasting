clear;
clc;

elevation=[];
residual=[];
time=[];

filenames=[];
dinfo = dir('O4_3_hourly_averaged\*.mat');

for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end

%t1 = datetime(1978,1,1,0,0,0);
%t2 = datetime(2018,11,31,23,0,0);
%t = t1:hours(1):t2;

for i=1:length(filenames)
    
    cd O4_3_hourly_averaged

    load (filenames(i))
    var=hour_vararray;

    time=[time;var(:,1)];
    elevation=[elevation;var(:,2)];
    residual=[residual;var(:,3)];
    cd ..
end
time=datetime(time,'ConvertFrom','datenum');
ntslf_tt_millport_surge=timetable(elevation, residual,'RowTimes',time);
ntslf_tt_millport_surge=retime(ntslf_tt_millport_surge,'hourly');
ntslf_tt_millport_surge=fillmissing(ntslf_tt_millport_surge,'linear','MaxGap',hours(1));

% save residual timetable
save('ntslf_tt_millport_surge.mat', "ntslf_tt_millport_surge")