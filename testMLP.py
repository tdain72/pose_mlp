import sys
import tensorflow as tf
import numpy as np

def GetSize(file):
    cnt = 0
    while True:
        line = file.readline().replace("\n", "").split(" ")
        if line[0] == '':
            break
        cnt += 1
    file.seek(0)
    return int(cnt/32)

def AssignJoints(skeleton, joint, frame, listArm):
    index = 0
    while True:
        for k in range(0, 32):
            line = skeleton.readline().replace("\n", "").split(" ")
            if '' == line[0]:
                break
            for a in listArm:
                if a == k:
                    for i in range(0,3):
                        joint[frame][index][i] = float(line[i])
                    index = (index + 1)%len(listArm)
        if '' == line[0]:
                break
        frame += 1
    skeleton.seek(0)
    return frame

def AssignJoints_a(skeleton, joint, frame):
    while True:
        for k in range(0, 32):
            line = skeleton.readline().replace("\n", "").split(" ")
            if '' == line[0]:
                break
            for i in range(0,8):
                joint[frame][k][i] = float(line[i])
        if '' == line[0]:
                break
        frame += 1
    skeleton.seek(0)
    return frame

if __name__ == "__main__":
    # Parameters
    learning_rate = 0.001
    batch_size = 100
    display_step = 1
    listArm = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

    # Network Parameters
    n_hidden_1 = 82 # 1st layer number of neurons
    n_hidden_2 = 64 # 2nd layer number of neurons
    n_hidden_3 = 32 # 3nd layer number of neurons
    n_hidden_4 = 64 # 4nd layer number of neurons
    n_hidden_5 = 82 # 5nd layer number of neurons
    n_input = len(listArm)*3
    n_classes = len(listArm)*3 

    # Read training data
    dataList = open("listInput.txt", "r")
    gtList = open("listGT.txt", "r")
    # dataList = open("extraData.txt", "r")
    # gtList = open("extraDataGT.txt", "r")
    gtFile = []
    dataFile = []

    for path in dataList:
        path = path.replace("\n", "")
        dataFile.append(open("input/"+path, "r"))

    for path in gtList:
        path = path.replace("\n", "")
        gtFile.append(open("gt/"+path, "r"))
    
    # for path in dataList:
    #     path = path.replace("\n", "")
    #     dataFile.append(open("input/Cut/"+path, "r"))

    # for path in gtList:
    #     path = path.replace("\n", "")
    #     gtFile.append(open("gt/Cut/"+path, "r"))

    gtSize = 0 
    dataSize = 0
    
    for files in dataFile:
        dataSize += GetSize(files)
    for files in gtFile:
        gtSize += GetSize(files)

    inputData = np.zeros((dataSize, len(listArm), 3))
    gtData = np.zeros((gtSize, len(listArm), 3))

    inputNum = 0
    gtNum = 0
    for files in dataFile:
        inputNum = AssignJoints(files, inputData, inputNum, listArm)
    for files in gtFile:
        gtNum = AssignJoints(files, gtData, gtNum, listArm)

    inputData = inputData.reshape(dataSize,-1)
    gtData = gtData.reshape(gtSize,-1)

     # Read test data
    # test_dataList = open("testdataList.txt", "r")
    # test_gtList = open("testGTList.txt", "r")
    # test_gtFile = []
    # test_dataFile = []

    # for path in test_dataList:
    #     path = path.replace("\n", "")
    #     test_dataFile.append(open("input/"+path, "r"))

    # for path in test_gtList:
    #     path = path.replace("\n", "")
    #     test_gtFile.append(open("gt/"+path, "r"))
    
    # test_gtSize = 0 
    # test_dataSize = 0
    
    # for files in test_dataFile:
    #     test_dataSize += GetSize(files)
    # for files in test_gtFile:
    #     test_gtSize += GetSize(files)

    # test_inputData = np.zeros((test_dataSize, len(listArm), 3))
    # test_gtData = np.zeros((test_gtSize, len(listArm), 3))

    # test_inputNum = 0
    # test_gtNum = 0
    # for files in test_dataFile:
    #     test_inputNum = AssignJoints(files, test_inputData, test_inputNum, listArm)
    # for files in test_gtFile:
    #     test_gtNum = AssignJoints(files, test_gtData, test_gtNum, listArm)
    
    # print(gtSize, dataSize)
    # test_inputData = test_inputData.reshape(test_dataSize,-1)
    # test_gtData = test_gtData.reshape(test_gtSize,-1)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    batch_prob = tf.placeholder(tf.bool)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
        'out': tf.Variable(tf.random_normal([n_hidden_5, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    # Create model
    def multilayer_perceptron(x):
        layer_1 = (tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_1 = tf.nn.relu(tf.layers.batch_normalization(layer_1,center = True, scale = True, training=batch_prob))

        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_2 = tf.nn.relu(tf.layers.batch_normalization(layer_2,center = True, scale = True, training=batch_prob))

        layer_3 = (tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        layer_3 = tf.nn.relu(tf.layers.batch_normalization(layer_3,center = True, scale = True, training=batch_prob))

        layer_4 = (tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
        layer_4 = tf.nn.relu(tf.layers.batch_normalization(layer_4,center = True, scale = True, training=batch_prob))

        layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
        # layer_5 = tf.layers.batch_normalization(layer_5,center = True, scale = True, training=batch_prob)

        out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.abs(Y-logits))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    batch_range = 100
    batch_size = int(dataSize/batch_range) - 1

    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, "modelNew1.ckpt")
        # print("Model restored")
        # Training cycle
        for epoch in range(10000):
            avg_cost = 0
            for i in range(0, batch_size):
                xData = inputData[batch_range*i:(i*batch_range+batch_range)]
                yData = gtData[batch_range*i:(i*batch_range+batch_range)]
                # Run optimization op (backprop) and cost op (to get loss value)
                result, c = sess.run([train_op, loss_op], feed_dict={X: xData, Y: yData, batch_prob: True})
                # Compute average loss
                avg_cost += c/batch_size
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            if(epoch % 1000 == 0):   
                saver.save(sess, "modelNew" + str(epoch) +".ckpt")
        print("Optimization Finished!")
        
        # # Test model
        # pred = tf.nn.softmax(logits)  # Apply softmax to logits
        # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print("Accuracy:", accuracy.eval({X: test_inputData, Y: test_gtData}))
    

    # test_output = np.zeros((test_dataSize,32,8))
    # outputFile = open("testOut.jnt","w")
    # num = 0
    # for files in test_dataFile:
    #     num = AssignJoints_a(files, test_output, num)

    # with tf.Session() as sess:
    #     saver.restore(sess, "modelNew1.ckpt")
    #     print("Model restored")

    #     result = sess.run(logits, feed_dict={X: test_inputData, batch_prob: True})
    #     result = np.reshape(result,(-1,len(listArm),3))
    #     print(result)

    #     for i in range(0,len(test_output[:,0,0])):
    #         dist = result[i,0,0:3] - test_output[i,0,0:3]
    #         r = 0
    #         for index in listArm:
    #             test_output[i,index,0:3] = result[i,r,:] - dist 
    #             r += 1
    
    # for i in range(0,len(test_output[:,0,0])):    
    #     for k in range(0,32):   
    #         data = "%f %f %f %f %f %f %d %d\n" % (test_output[i,k,0],test_output[i,k,1],test_output[i,k,2],test_output[i,k,3],test_output[i,k,4],test_output[i,k,5],test_output[i,k,6],test_output[i,k,7])
    #         outputFile.write(data)