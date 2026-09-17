def coinChange(coins, amount):
    dp = [amount + 1] * (amount + 1)
    print(dp)
    dp[0] = 0

    for current in range(1, amount + 1):        
        for coin in coins:
            if coin <= current:
                dp[current] = min(
                    dp[current],
                    dp[current - coin] + 1
                )
            print(dp)

    return -1 if dp[amount] == amount + 1 else dp[amount]


coins = [1, 2, 5]
amount = 11

print(coinChange(coins, amount))
