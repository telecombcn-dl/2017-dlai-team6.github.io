delimiterIn = '\n';

%%
figure
filename = 'pacman2.txt';
[A,delimiterOut]  = importdata(filename,delimiterIn);
episode = linspace(0,120,length(A))';
plot(episode,A)
title(filename)
xlabel('episode')
ylabel('reward')
%%
figure
filename = 'pacman4.txt';
[A,delimiterOut]  = importdata(filename,delimiterIn);
episode = linspace(0,120,length(A))';
plot(episode,A)
title(filename)
xlabel('episode')
ylabel('reward')
%%
figure
filename = 'pacman6.txt';
[A,delimiterOut]  = importdata(filename,delimiterIn);
episode = linspace(0,120,length(A))';
plot(episode,A)
title(filename)
xlabel('episode')
ylabel('reward')