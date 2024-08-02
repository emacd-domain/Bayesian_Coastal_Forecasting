worldmap([30 75],[-40 25])
load coastlines
plotm(coastlat,coastlon,'LineWidth',2,'Color','k')

load domainlocs

geoshow(domainlocs(:,2),domainlocs(:,1),'DisplayType','point','Color','b','MarkerSize',1,'Marker','*','MarkerEdgeColor', 'auto')
plotm(coastlat,coastlon,'LineWidth',2,'Color','k')

