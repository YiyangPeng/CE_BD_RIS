function P = func_P(MG)

% [groupSize, ~] = size(Y);
num = MG*(MG + 1)/2;
P = zeros(MG^2,num);

for i = 1 : MG

    for j = 1 : MG
        if i >= j
            P(MG*(i - 1) + j, i + (2*MG - j)*(j - 1)/2) = 1;
        else
            P(MG*(i - 1) + j, j + (2*MG - i)*(i - 1)/2) = 1;
        end
    end
end

end

