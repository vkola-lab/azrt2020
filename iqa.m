
NIQE = [];
BRIS = [];

for i=0:10:1990
    filename = sprintf('./output/%d.png', i);
    img = imread(filename);
    NIQE = [NIQE niqe(img)];
    BRIS = [BRIS brisque(img)];
end

plot(NIQE);