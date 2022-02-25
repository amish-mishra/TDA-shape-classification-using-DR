import gudhi

diag1 = [[2.7, 3.7],[9.6, 14.],[34.2, 34.974], [3.,float('Inf')]]
diag2 = [[2.8, 4.45],[9.5, 14.1],[3.2,float('Inf')]]

message = "Bottleneck distance approximation = " + '%.2f' % gudhi.bottleneck_distance(diag1, diag2, 0.1)
print(message)

message = "Bottleneck distance value = " + '%.2f' % gudhi.bottleneck_distance(diag1, diag2)
print(message)