function flatArr = flatten(arr)
    res = zeros(1, 64);
    zigzag = readmatrix("Zig-Zag Pattern.txt");
    for i = 1:size(zigzag, 1)
        for j = 1:size(zigzag, 2)
            idx = zigzag(i, j)  + 1;
            val = arr(i, j);
            res(1, idx) = val;
        end 
    end
    flatArr = res;
end
