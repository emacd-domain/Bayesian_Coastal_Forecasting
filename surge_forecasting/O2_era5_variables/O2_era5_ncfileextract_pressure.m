% this file will extract the points from netcdffiles, filter them based
% on the fetch radius and retime them
load ..\O1_location_selection\domainlocs.mat

clearvars -except domainlocs

%load pointlatlon
windlon=domainlocs(:,1);
windlat=domainlocs(:,2);

cd I:\CivilEng\Personal\dsb14145
filenames=[];
dinfo = dir('era-5_pressure_1980-2020\*.nc');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end
cd era-5_pressure_1980-2020


cd I:\CivilEng\Personal\dsb14145\era-5_pressure_1980-2020
MSL=[];
%V10=[];
DT=[];
for j=41:length(filenames)
    cd I:\CivilEng\Personal\dsb14145\era-5_pressure_1980-2020
    tic
    j

    %%% all lats and longs to create target array
    %eralatall = ncread(filenames(1),'latitude');
    %eralonall = ncread(filenames(1),'longitude');
    %%% stride lat and long every 2.5 degrees lat and lon
    eralat = ncread(filenames(j),'latitude',[1],[inf],[1]);
    eralon = ncread(filenames(j),'longitude',[1],[inf],[1]);
    [geralon,geralat]=meshgrid(eralon,eralat);
    s=size(geralon);
    x=reshape(geralon,[s(1)*s(2),1]);
    y=reshape(geralat,[s(1)*s(2),1]);

    dtime=double(ncread(filenames(j),'time'));
    mslord=zeros(length(dtime),434);
    msl=ncread(filenames(j),'msl',[1,1,1],[inf,inf,inf],[1 1 1]);
for i=1:length(windlon)
    
    lonindex = find(eralon == windlon(i));
    latindex = find(eralat == windlat(i));
    
    mslord(:,i)=squeeze(msl(lonindex,latindex,:));
    %v10=ncread(filenames(j),'v10',[1,1,1],[inf,inf,inf],[2 2 1]);

end
    dtime=datetime([1900,1,1,0,0,0])+hours(dtime);
    MSL=timetable(dtime,mslord);
    MSL=retime(MSL,'hourly');
    cd C:\Users\alphonse\Desktop\forecast_surge_metabuild\O2_era5_variables\msl_pressure_at_domain_locs
    save("msl_"+string(1979+j)+".mat","MSL");
    clearvars -except filenames domainlocs windlon windlat eralon eralat j
    toc
end


