from networks import CNN

cnn = CNN('./configuration.json', 1000)
valid_optimal_accu = cnn.train()
test_accu = cnn.test()




