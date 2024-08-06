# 定义一个字符串
input_string = "be9d4056 b8f1c364 347a4212 bc6f409e 3600c12a b946bd55 3f183d91 3d86ba06 40ef3ea3 c456ad00 3c223d56 c354c31f c07ec0c9 3f5ebb28 ad10be8e 3f85bac9 c41a401a bc5ec275 bf08c0a2 45864517 3cc2c256 407d3f85 bfd63d14 be5bbdb0 3f64bd70 bd67b95a 3e5e41ce c42ac354 39b83f39 be9a3bdd b950bcb4 372a40ac"

# 按照空格分隔字符串，并将结果存入数组
words = input_string.split(' ')

# 将数组倒序
reversed_words = words[::-1]

# 打印倒序后的数组，字符间以空格分块
for word in reversed_words:
    print(word, end =' ')  # 使用end=' '来在打印后不换行，而是在单词后添加空格
print()  # 最后打印一个换行符，以确保输出格式正确