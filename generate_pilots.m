function X = generate_pilots(K, U, L)
    % Generate (K*U) x L pilot matrix with:
    % - |X[i,j]| = 1
    % - X_k X_k^H = L*I_U (antenna orthogonality)
    % - X_k X_k'^H = 0 (user orthogonality)
    
    % Input validation
    if mod(L, U) ~= 0
        error('L must be multiple of U for this design');
    end
    if mod(L, K) ~= 0
        error('L must be multiple of K for this design');
    end
    if mod(L, K*U) ~= 0
        error('L must be multiple of K*U for this design');
    end

    % Total number of orthogonal sequences needed
    N = K * U;

    % Create N orthogonal sequences of length L using DFT
    n = 0 : N-1; 
    k = n';
    X = exp(-1j * 2 * pi * n .* k / N);

    % If L > N, repeat columns while maintaining orthogonality
    if L > N
        reps = floor(L / N);
        remainder = mod(L, N);
        X = repmat(X, [1, reps]);
        if remainder > 0
            X = [X, X(:, 1:remainder)];
        end
    end
end

