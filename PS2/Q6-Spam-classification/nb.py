import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    X1, X0 = matrix[category==1], matrix[category==0]
    state['phi_js_yeq1'] = (np.sum(X1, axis=0) + 1) / (np.sum(X1) + N)
    state['phi_js_yeq0'] = (np.sum(X0, axis=0) + 1) / (np.sum(X0) + N)
    state['phi_y'] = np.sum(category==1) / category.shape[0]
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    phi_js_yeq1, phi_js_yeq0, phi_y = state['phi_js_yeq1'], state['phi_js_yeq0'], state['phi_y']

    L1 = np.log(phi_js_yeq1) * matrix # (n,) * (matrix.shape[0], n) = (matrix.shape[0], n)
    L0 = np.log(phi_js_yeq0) * matrix
    log_phi_yeq1 = np.sum(L1, axis=1) # (matrix.shape[0],)
    log_phi_yeq0 = np.sum(L0, axis=1)
    r = log_phi_yeq0 + np.log(1-phi_y) - log_phi_yeq1 - np.log(phi_y)
    probs = 1./(1 + np.exp(np.clip(r, -700, 700)))
    
    output[probs>.5] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
