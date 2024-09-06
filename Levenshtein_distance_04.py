def levenshtein_distance(str1, str2, cost_insert=1, cost_delete=1, cost_substitute=1,pairwise_swap=1):
    """
    Find the distance between two strings, defined as the total number of
    single-symbol operations that lead from one string to another. These
    operations include deletion,insertion,substitution and the pairwise swapping of
    two subsequent symbols

    Args:
        str1 (str): sequence of symbols #1
        str2 (str): sequence of symbols #2
        cost_insert (int): custom cost of insertion
        cost_delete (int): custom cost of deletion
        cost_substitute (int): custom cost of substitution
        pairwise_swap (int): custom cost of swapping two neighbouring symbols

    Returns:
        int: the Levenshtein distance.
    """

    # THE BASIC SOLUTION INCLUDES BUILDING UP THE MATRIX D(i,j)
    # THE FIRST ROW: D(0, j) = D(0, j - 1) + COST_INSERT S2[j]
    # THE FIRST COLUMN: D(i, 0) = D(i - 1, 0) + COST_DELETE S1[i]
    # ALL OTHER CELLS:
    # D(i, j) = min{
    # D(i - 1, j) + COST_DELETE S1[i],
    # D(i, j - 1) + COST_INSERT S2[j],
    # D(i - 1, j - 1) + COST_SUBSTITUTE S1[i] for S2[j]
    # D(i - 2, j - 2) + COST_SWAP
    # }
    # THE LEVENSHTEIN DISTANCE VALUE IS AT THE BOTTOM RIGHT CELL

    len1, len2 = len(str1), len(str2)

    # Create a 2D array to store distances
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i * cost_delete
    for j in range(len2 + 1):
        dp[0][j] = j * cost_insert

    # Compute distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = cost_substitute
            dp[i][j] = min(
                dp[i - 1][j] + cost_delete,  # Deletion
                dp[i][j - 1] + cost_insert,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
                dp[i - 2][j - 2] + pairwise_swap #pairwise swap
            )

    return dp[len1][len2]

print(levenshtein_distance("apple", "aptle"))