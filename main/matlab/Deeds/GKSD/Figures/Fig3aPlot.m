load 3a
lw = 3;
hold all

figure(1)
plot(rvec,y,'k:','Linewidth',lw);
ax = gca;
set(ax,'FontSize',16,'FontWeight','bold','LineWidth',3,...
    'PlotBoxAspectRatio',[1.79/1.25 1 1],'XMinorTick','on',...
    'YMinorTick','on','XScale','log','YScale','log');
ax.XAxis.TickLength = [0.025 0.0375];
ax.YAxis.TickLength = [0.025 0.0375];
xlabel('r','FontWeight','bold','FontSize',20);
ylabel('Total Substrate [S]_T [nM]','FontWeight','bold','FontSize',20,...
    'Units','Normalized','Position',[-0.08, 0.5, 0]);
% line([1e2,1e2],[1e2,1e7],'Color','red','Linewidth',4);
box off
annotation(figure(1),'arrow',[0.758928571428571 0.807142857142857],...
    [0.815666666666668 0.869047619047621],'LineWidth',3);
annotation(figure(1),'textbox',...
    [0.790285714285714 0.771428571428571 0.0900714285714289 0.0907142857142876],...
    'String',{'Larger','    Q'},...
    'FontWeight','bold',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor','none');
hold off