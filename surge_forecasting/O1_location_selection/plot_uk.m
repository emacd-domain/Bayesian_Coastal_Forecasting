load coastlines
cd circlem_v1/circlem

% large 1000 km circle, low resolution
[~,clat0,clon0]=circlem(55.75,-4.9,1000);
%glon0=-20:1:25;glat0=30:1:75;[X0,Y0]=meshgrid(glon0,glat0);
%s=size(X0);x0=reshape(X0,[s(1)*s(2),1]);y0=reshape(Y0,[s(1)*s(2),1]);
%index0=inpolygon(x0,y0,clon0,clat0);
%ind0=find(index0==1);x0=x0(ind0);y0=y0(ind0);

% 400 km cicle medium resolution
%[~,clat1,clon1]=circlem(55.75,-4.9,400);
%glon1=-20:0.5:25;glat1=30:0.5:75;[X1,Y1]=meshgrid(glon1,glat1);
%s=size(X1);x1=reshape(X1,[s(1)*s(2),1]);y1=reshape(Y1,[s(1)*s(2),1]);
%index1=inpolygon(x1,y1,clon1,clat1);
%ind1=find(index1==1);x1=x1(ind1);y1=y1(ind1);

% 200 km circle high resolution
%[~,clat2,clon2]=circlem(55.75,-4.9,200);
clat2 = [62.5, 62.5, 47.5, 47.5];
clon2 = [-15, 5, 5, -15];
glon2=-20:0.5:25;glat2=30:0.5:75;[X2,Y2]=meshgrid(glon2,glat2);
s=size(X2);x2=reshape(X2,[s(1)*s(2),1]);y2=reshape(Y2,[s(1)*s(2),1]);
index2=inpolygon(x2,y2, clon2,clat2);
ind2=find(index2==1);x2=x2(ind2);y2=y2(ind2);

% commbine all and remove repeats
locs=[x2,y2];
locs=unique(locs,'rows');

% make UK and Ireland island polygon
unitedkingdom = 7981:8087;
uk_poly = polyshape(coastlon(unitedkingdom),coastlat(unitedkingdom));
ireland = 8088:8117;
ire_poly = polyshape(coastlon(ireland),coastlat(ireland));

% leave points outwith polygons
index=~inpolygon(locs(:,1), locs(:,2), uk_poly.Vertices(:,1), uk_poly.Vertices(:,2));
ind=find(index==1);locs=locs(ind,:);
index=~inpolygon(locs(:,1), locs(:,2), ire_poly.Vertices(:,1), ire_poly.Vertices(:,2));
ind=find(index==1);locs=locs(ind,:);

% remove east coast points
% euro coast
removepoints = [flip(uk_poly.Vertices(60:90,:),1);[clon0(10:47),clat0(10:47)];[coastlon(5058:5083),coastlat(5058:5083)];uk_poly.Vertices(90,:)];
index=~inpolygon(locs(:,1), locs(:,2), removepoints(:,1), removepoints(:,2));
ind=find(index==1);locs=locs(ind,:);
ind=find((locs(:,2)<50)&(locs(:,1)>0));locs(ind,:)=[];;


%figure
%plot(removepoints(:,1), removepoints(:,2),'k')

% plot UK
%plot(uk_poly)
%hold on
%plot(ire_poly)

worldmap([44 66],[-30 10])
geoshow(locs(:,2),locs(:,1),'DisplayType','point','Color','b','MarkerSize',4,'Marker','.','MarkerEdgeColor', 'auto')
%geoshow(clat0,clon0,'DisplayType','line','Color', 'r','LineStyle',':','LineWidth',1)
%geoshow(clat1,clon1,'DisplayType','line','Color', 'r','LineStyle',':','LineWidth',1)
%geoshow(clat2,clon2,'DisplayType','line','Color', 'r','LineStyle',':','LineWidth',1)
plotm(coastlat,coastlon,'k')
