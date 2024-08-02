% this file will extract the points from netcdffiles, filter them based
% on the fetch radius and retime them
clear;clc;

cd I:/CMEMS-EuropeanShelf

lat_id = 174;
lon_id = 416;

filenames=[];
dinfo = dir('european_shelf_databin/*.nc');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end

allregrid=[];

hm0=[];
tm0=[];
mdr=[];
dt=[];

for i=1:length(filenames)
       i
       A=ncread("european_shelf_databin/"+filenames(i),'VHM0_WW',[lat_id lon_id 1],[1 1 inf]);
       C=ncread("european_shelf_databin/"+filenames(i),'VTM01_WW',[lat_id lon_id 1],[1 1 inf]);
       D=ncread("european_shelf_databin/"+filenames(i),'VMDR_WW',[lat_id lon_id 1],[1 1 inf]);
       B=ncread("european_shelf_databin/"+filenames(i),'time');
       hm0=cat(3,hm0,A);
       tm0=cat(3,tm0,C);
       mdr=cat(3,mdr,D);
       dt=cat(3,dt,B');
end

dtr=[];
for j = 1:length(filenames)
    dtr=[dtr,squeeze(dt(:,:,j))];
end

dtr=dtr';
mdr=squeeze(mdr);
tm0=squeeze(tm0);
hm0=squeeze(hm0);

wavedata = timetable(dt, hm0, tm0, mdr);

wavedata = retime(wavedata,'hourly','linear');

cmems_wavechar_model = wavedata(1:350640,:);
cmems_wavechar_vc_test = wavedata(350641:end,:);

save('cmems_wavechar_model.mat','cmems_wavechar_model')
save('cmems_wavechar_vc_test.mat','cmems_wavechar_vc_test')