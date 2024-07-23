worldmap([30 75],[-40 25])
load coastlines
plotm(coastlat,coastlon,'LineWidth',2,'Color','k')
cd circlem_v1/circlem

toplon = [-6.17,-6.75];toplat = [55.75,55.17];
toplonlin = linspace(toplon(1),toplon(2),11);
toplatlin = linspace(toplat(1),toplat(2),11);

botlon = [-6.33,-5.17];botlat = [52.17,51.83];
botlonlin = linspace(botlon(1),botlon(2),11);
botlatlin = linspace(botlat(1),botlat(2),11);

hold on
[~,circlelattop,circlelontop]=circlem(mean(toplatlin),mean(toplonlin),1600);
[~,circlelatbot,circlelonbot]=circlem(mean(botlatlin),mean(botlonlin),1600);
cd ../..

geoshow(toplatlin,toplonlin,'DisplayType','line','Color', 'b')
geoshow(botlatlin,botlonlin,'DisplayType','line','Color', 'b')

botbound=[];
for i = 6%:9
    i
    for  j = 1:1:100
        dist=[];
        vec = [botlonlin(i),circlelonbot(j);botlatlin(i),circlelatbot(j)];
        p = InterX(vec,[coastlon';coastlat']);
        dist=sqrt((botlonlin(i)-p(1,:)).^2+((botlatlin(i)-p(2,:)).^2));
        [~,index]=min(dist);
        if ~isempty(p)
            vec(1,2)=p(1,index); vec(2,2)=p(2,index);
        end
        geoshow(vec(2,:),vec(1,:),'DisplayType','line','Color', 'r')
        botbound(1,j)=vec(1,2);
        botbound(2,j)=vec(2,2);
    end
end

topbound=[];
for i = 6%2:9
    i
    for  j = 1:1:100
        dist=[];-*
        vec = [toplonlin(i),circlelontop(j);toplatlin(i),circlelattop(j)];
        p = InterX(vec,[coastlon';coastlat']);
        dist=sqrt((toplonlin(i)-p(1,:)).^2+((toplatlin(i)-p(2,:)).^2));
        [~,index]=min(dist);
        if ~isempty(p)
            vec(1,2)=p(1,index); vec(2,2)=p(2,index);
        end
        geoshow(vec(2,:),vec(1,:),'DisplayType','line','Color', 'b')
        topbound(1,j)=vec(1,2);
        topbound(2,j)=vec(2,2);
    end
end
s=size(X);
plotm(coastlat,coastlon,'LineWidth',2,'Color','k')
% geoshow(bound(2,:),bound(1,:),'DisplayType','line','Color', 'k')
geoshow(55.75,-5,'DisplayType','point','MarkerSize',10,'Marker','*','Color', 'r','MarkerEdgeColor', 'auto')

t = polyshape(topbound(1,:)',topbound(2,:)');
b = polyshape(botbound(1,:)',botbound(2,:)');

all=union(b,t);

locbox=[54.7,55.7,55.7,54.7;-6,-6,-4.7,-4.7];
focpoly = polyshape(locbox(2,:),locbox(1,:));

all=union(all,focpoly);

glon1=-50:1:25;glat1=30:1:75;[X1,Y1]=meshgrid(glon1,glat1);
s=size(X1);x1=reshape(X1,[s(1)*s(2),1]);y1=reshape(Y1,[s(1)*s(2),1]);

glon2=-50:0.5:25;glat2=30:0.5:75;[X2,Y2]=meshgrid(glon2,glat2);
s=size(X2);x2=reshape(X2,[s(1)*s(2),1]);y2=reshape(Y2,[s(1)*s(2),1]);

glon3=-50:0.25:25;glat3=30:0.25:75;[X3,Y3]=meshgrid(glon3,glat3);
s=size(X3);x3=reshape(X3,[s(1)*s(2),1]);y3=reshape(Y3,[s(1)*s(2),1]);
cd circlem_v1/circlem
[~,clat1,clon1]=circlem(55.75,-4.9,500);
[~,clat2,clon2]=circlem(55.75,-4.9,250);

index1=inpolygon(x1,y1,all.Vertices(:,1),all.Vertices(:,2));
index2=inpolygon(x2,y2,clon1,clat1);index2a=inpolygon(x2,y2,all.Vertices(:,1),all.Vertices(:,2));
index3=inpolygon(x3,y3,clon2,clat2);index3a=inpolygon(x3,y3,all.Vertices(:,1),all.Vertices(:,2));

ind=find(index1==1);x1=x1(ind);y1=y1(ind);
ind=find((index2==1)&(index2a==1));x2=x2(ind);y2=y2(ind);
ind=find((index3==1)&(index3a==1));x3=x3(ind);y3=y3(ind);

locs=[[x1,y1];[x2,y2];[x3,y3]];
domainlocs=unique(locs,'rows');

cd ..\..

save('domainlocs.mat', 'domainlocs')

geoshow(locs(:,2),locs(:,1),'DisplayType','point','Color','b','MarkerSize',1,'Marker','*','MarkerEdgeColor', 'auto')
plotm(coastlat,coastlon,'LineWidth',2,'Color','k')

