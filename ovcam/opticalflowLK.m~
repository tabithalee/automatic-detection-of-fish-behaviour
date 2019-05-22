vidReader = VideoReader('BC_POD1_PTILTVIDEO_20110527T063643.000Z_1.ogg');
opticFlow = opticalFlowLK('NoiseThreshold', 0.009);
h = figure;
%btn = uibutton(h);

%set(h,'MenuBar','none','Units','Normalized');
%set(h,'Position',[0.5 0.5 0.05 0.05]);

%movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

 c1 = uicontrol(... % Button for pausing selected plot
    'Enable', 'on', ...
    'Parent', hViewPanel, ...
    'Units','pixels',...
    'HandleVisibility','callback', ...
    'String','Pause',...
    'BackgroundColor',[1 0.6 0.6], ...
    'Position', [10, 10, 50, 20]);

c1.Callback = @pauseButtonPushed;

 c2 = uicontrol(... % Button for playing selected plot
    'Enable', 'on', ...
    'Parent', hViewPanel, ...
    'Units','pixels',...
    'HandleVisibility','callback', ...
    'String','Play',...
    'BackgroundColor',[1 0.6 0.6], ...
    'Position', [70, 10, 40, 20]);

c2.Callback = @playButtonPushed;

c3 = uicontrol(... % Button for stepping selected plot
    'Enable', 'on', ...
    'Parent', hViewPanel, ...
    'Units','pixels',...
    'HandleVisibility','callback', ...
    'String','Step',...
    'BackgroundColor',[1 0.6 0.6], ...
    'Position', [120, 10, 40, 20]);

c3.Callback = {@frameAndPlot, {@readFrame}};

align([c1 c2], 'distributed', 'bottom');

while hasFrame(vidReader)
    pause(0.01)
    frameAndPlot(vidReader, opticFlow, hPlot);
    pause(0.02)
end


close(h);

function pauseButtonPushed(src,event)
    fprintf('paused\n');
    uiwait(gcf);
end

function playButtonPushed(src,event)
    fprintf('playing\n');
    uiresume(gcbf);
end

function frameAndPlot(vidReader, opticFlow, hPlot)
    frameRGB = readFrame(vidReader);
    frameGray = rgb2gray(frameRGB);
    flow = estimateFlow(opticFlow,frameGray);
    imshow(frameRGB)
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',hPlot);
    hold off
end