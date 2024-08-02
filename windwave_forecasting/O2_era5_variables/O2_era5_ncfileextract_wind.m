% this file will extract the points from netcdffiles, filter them based
% on the fetch radius and retime them
load ..\O1_location_selection\domainlocs.mat

currentFolder = pwd

clearvars -except domainlocs

%load pointlatlon
windlon=domainlocs(:,1);
windlat=domainlocs(:,2);

cd I:\
filenames=[];
dinfo = dir('era-5_wind_1980-2020\*.nc');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end

U10=[];
V10=[];
DT=[];

for j=1:length(filenames)
    cd I:\era-5_wind_1980-2020
    tic
    j

    %%% all lats and longs to create target array
    %eralatall = ncread(filenames(1),'latitude');
    %eralonall = ncread(filenames(1),'longitude');
    %%% stride lat and long every 2.5 degrees lat and lon

    eralat = ncread(filenames(j),'latitude', [1], [inf], [1]);
    eralon = ncread(filenames(j),'longitude', [1], [inf], [1]);
    [geralon,geralat]=meshgrid(eralon,eralat);
    s=size(geralon);
    x=reshape(geralon,[s(1)*s(2),1]);
    y=reshape(geralat,[s(1)*s(2),1]);

    dtime=double(ncread(filenames(j),'time'));
    u10ord=zeros(length(dtime),434);
    v10ord=zeros(length(dtime),434);
    
    u10=ncread(filenames(j),'u10',[1,1,1],[inf, inf, inf],[1 1 1]);
    v10=ncread(filenames(j),'v10',[1,1,1],[inf, inf, inf],[1 1 1]);

for i=1:length(windlon)
    
    lonindex = find(eralon == windlon(i));
    latindex = find(eralat == windlat(i));
    
    u10ord(:,i)=squeeze(u10(lonindex,latindex,:));
    v10ord(:,i)=squeeze(v10(lonindex,latindex,:));
    
end
    
    dtime=datetime([1900,1,1,0,0,0])+hours(dtime);

    U10=timetable(dtime,u10ord);
    U10=retime(U10,'hourly');
    
    V10=timetable(dtime,v10ord);
    V10=retime(V10,'hourly');
    
    cd (currentFolder)
    cd O2_era5_variables\u10_wind_at_domain_locs
    save("u10_"+string(1979+j)+".mat","U10");

    cd (currentFolder)
    cd O2_era5_variables\v10_wind_at_domain_locs
    save("v10_"+string(1979+j)+".mat","V10");

    clearvars -except filenames domainlocs windlon windlat eralon eralat j

    toc
end


