
import numpy as np
import pandas as pd 


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    # use a dictionary to store the index of each element in the list
    unique_dict = {}
    for i in range(len(unique_list)):
        unique_dict[unique_list[i]] = i 
    return unique_list, unique_dict

def convert_DF_to_RDD(DF):
    """ 
    This function converts a rating dataframe into a dictionary of lists
    Args:
        DF: a dataframe which contains 'user', 'item' and 'rating' columns
    Returns:
        RDD: a dictionary which contains 
            'total': the total number of rating
            'users': a list of user id for each rating
            'items': a list of item id for each rating
            'ratings': a list of ratings
            'user_indices': a dictionary mapping each unique user ID to its corresponding index in the feature matrix
            'item_indices': a dictionary mapping each unique item ID to its corresponding index in the feature matrix
    """ 
    RDD = {'total': 0, 'users': [], 'items': [], 'ratings': [], 'user_indices': {}, 'item_indices': {}}
    
    # Get the total number of ratings
    RDD['total'] = DF.shape[0]
    
    # Extract the values for the 'user', 'item', and 'rating' columns
    RDD['users'] = DF['user'].values
    RDD['items'] = DF['item'].values
    RDD['ratings'] = DF['rating'].values
    
    # Get the unique user and item IDs and their corresponding indices in the feature matrix
    unique_users, user_indices = unique(RDD['users'])
    unique_items, item_indices = unique(RDD['items'])
    
    RDD['user_indices'] = user_indices
    RDD['item_indices'] = item_indices
    
    return RDD

# assume now you have obtained trainRDD and testRDD
# Compute the objective funtion

def computeMSE(RDD,P,Q,la=0):
    """ 
    This function computes regularized Mean Squared Error (MSE)
    Args:
        RDD: a dict of list of userID, itemID, Rating
        P: user's features matrix (M by K)
        Q: item's features matrix (N by K)
        la: lambda parameter for regulization (see definition in the project description)
    Returns:
        mse: mean squared error
    """ 
    
    # Get the number of ratings and the lists of user IDs, item IDs, and ratings from the RDD
    N = RDD['total']
    users = RDD['users']
    items = RDD['items']
    ratings = RDD['ratings']
    unique_items, items_dict =unique(items)
    users_items, users_dict =unique(users)
    
    # Initialize the squared error sum to 0
    squared_error_sum = 0
    
    # Iterate over the ratings
    for i in range(N):
        # Get the user ID and item ID for the current rating
        u = users_dict[users[i]]
        j = items_dict[items[i]]
        
        # Get the user and item feature vectors
        p = P[u]
        q = Q[j]
        
        # Compute the error for the current rating
        r = ratings[i]
        error = r - np.dot(p, q)
        
        # Add the squared error to the sum
        squared_error_sum += error ** 2
        
        # Regularize the error by adding the penalty term
        squared_error_sum += la * (np.sum(np.where(p<0, p, 0)** 2 + np.where(q<0, q, 0)** 2))
    
    # Compute the MSE by dividing the squared error sum by the number of ratings
    mse = squared_error_sum / N
    
    return mse    
 
    

def line_search(RDD, P, Q, dP, dQ, LAMBDA, GAMMA, mse):
    # Update P and Q using the step size and gradients
    P_new = P - GAMMA * dP
    Q_new = Q - GAMMA * dQ

    # Calculate the MSE for the updated matrices
    mse_new = computeMSE(RDD, P_new, Q_new, LAMBDA)

    # If the MSE has decreased, return the updated matrices and step size
    if mse_new < mse:
        return P_new, Q_new, GAMMA, mse_new

    # If the MSE has not decreased, reduce the step size and try again
    else:
        GAMMA /= 2
        
        # If the step size is too small, return the original matrices and step size
        if GAMMA < 1e-8:
            return P, Q, GAMMA, mse
        
        # Otherwise, continue the loop
        else:
            return line_search(RDD, P, Q, dP, dQ, LAMBDA, GAMMA, mse)

    



# Compute the gradient of the objective funtion
def computeGradMSE(RDD, P, Q, la=0):
    """ 
    This function computes the gradient of regularized MSE with respect to P and Q
    Args:
        RDD: a dict of list of userID, itemID, Rating
        P: user's features matrix (M by K)
        Q: item's features matrix (N by K)
        la: lambda parameter for regularization (see definition in the project description)
    Returns:
        gradP, gradQ: gradient of mse with respect to each element of P and Q 
    """
    
    # Get the number of ratings and the lists of user IDs, item IDs, and ratings from the RDD
    N = RDD['total']
    users = RDD['users']
    items = RDD['items']
    ratings = RDD['ratings']
    unique_items, items_dict = unique(items)
    users_items, users_dict = unique(users)
    
    # Get the number of users and items
    m, n = P.shape[0], Q.shape[0]
    
    # Get the number of features
    k = P.shape[1]
    
    # Initialize the gradient matrices to zero
    gradP = np.zeros((m, k))
    gradQ = np.zeros((n, k))
    
    for i in range(N):
        # Get the user ID and item ID for the current rating
        u = users_dict[users[i]]
        j = items_dict[items[i]]
        
        # Get the user and item feature vectors
        p = P[u]
        q = Q[j]
        
        # Compute the error for the current rating
        e = ratings[i] - np.dot(p, q)
        
        # Update the gradient matrices
        gradP[u, :] += -2 * e * q + 2 * la * h(p)
        gradQ[j, :] += -2 * e * p + 2 * la * h(q)
        
    return gradP/ N, gradQ/ N

def h(x):
    return np.where(x < 0, x, 0)


    


def GD(RDD,M,N,K,MAXITER=50, GAMMA=0.001, LAMBDA=0.05, adaptive=0):
    """ 
    This function implemnts the gradient-descent method to minimize the regularized MSE with respect to P and Q
    Args:
        RDD: a dict of list of users, items, ratings
        M: number of users
        N: number of items
        K: rank parameter
        MAXITER: maximal number of iterations (epoches) of GD 
        GAMMA: step size of GD
        LAMBDA: regulization parameter lambda in the mse loss
        adaptive: if 0 then use constant step size GD, 
                  if 1 then use line search to choose the step size automatically
    Returns:
        P: optimal P found by GD
        Q: optimal Q found by GD
        lreg_mse: a list of regulized mse values evaluated on RDD, after each iteration
        lmse: a list of mse values, evaluated on RDD after each iteration
        other scores for analysis purpose
    """ 
    
    # Initialize the user and item feature matrices to random values
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K) 
    """K = 14 # rank parameter (best chose)
    P = np.zeros((M,K)) # user's features matrix (M by K)
    Q = np.zeros((N,K)) # item's features matrix (N by K)  """
    # Initialize the lists to store the MSE and regularized MSE after each iteration
    lreg_mse = []
    lmse = []
    
    # Initialize the iteration counter
    iter = 0
    mse = 100
    reg_mse =100
    
    # Run the gradient descent loop
    while iter < MAXITER:
        # Compute the gradients ofthe regularized MSE loss with respect to P and Q
        dP, dQ = computeGradMSE(RDD, P, Q, LAMBDA)
        # Update P and Q using gradient descent
        if adaptive == 0:
            # Use constant step size
            P = P - GAMMA * dP
            Q = Q - GAMMA * dQ
            reg_mse = computeMSE(RDD, P, Q, LAMBDA )
        else:
            # Use line search to find the optimal step size
            P, Q, GAMMA, reg_mse = line_search(RDD, P, Q, dP, dQ, LAMBDA, GAMMA, mse)

        # Compute the regularized MSE and MSE on the training set
        mse = computeMSE(RDD, P, Q)
        
        # Append the errors to the lists
        lreg_mse.append(reg_mse)
        lmse.append(mse)

        # Increment the iteration counter
        iter += 1

    return P, Q, lreg_mse, lmse