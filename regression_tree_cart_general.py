import scipy
import scipy.optimize
import numpy
import copy
import sys
import pygame
import time
import random
import math
from bisect import bisect_right
from operator import itemgetter
r = 1
l = 2
BLACK = ( 0, 0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
pygame.init()
ft_size = 25
tree_font = pygame.font.SysFont("Arial", ft_size)

class Tree(object):
    def __init__(self, predict, stdev, start, num_points):
        self.error = None
        self.predict = predict
        self.stdev = stdev
        self.start =  start
        self.split_var = None
        self.split_val = None
        self.split_lab = None
        self.left = None
        self.right = None
        self.num_points = num_points
        self.left_wt = None
        self.right_wt = None
        self.morta = None
        self.internal_nodes = None
        
    def lookup(self, x):
        """Returns the predicted value given the parameters."""
        if self.left == None:
            return self.predict
        if x[self.split_var] <= self.split_val:
            return self.left.lookup(x)
        return self.right.lookup(x)

    def lookup_df(self, df):
        """Run through the built tree"""
        if self.right == None:
            return 0.0, 0.0, 0

        dfs = df[df['var'] == self.split_lab]

        id1 = list(dfs[dfs['val'] > self.split_val]['pid_contrb'])

        dfs1 = df[df['pid_contrb'].isin(id1)]

        dfs2 = df[-df['pid_contrb'].isin(id1)]


        ctb1sum = dfs1[dfs1['var'] == self.split_lab]['ctb_contrb'].mean()

        if len(numpy.array(dfs1[dfs1['var'] == self.split_lab]['ctb_contrb'])) == 0:
            ctb1sum = 0.0

        ctb2sum = dfs2[dfs2['var'] == self.split_lab]['ctb_contrb'].mean()

        if len(numpy.array(dfs2[dfs2['var'] == self.split_lab]['ctb_contrb'])) == 0:
            ctb2sum = 0.0

        node_ctb = ctb2sum - ctb1sum

        df_l = dfs2
        df_r = dfs1


        left_avg, left_sum, left_itn = self.left.lookup_df(df_l)
        right_avg, right_sum, right_itn = self.right.lookup_df(df_r)

        cur_sum = - numpy.sign(self.left_wt - self.right_wt) * node_ctb + left_sum +right_sum
        cur_itn = 1 + left_itn + right_itn
        cur_avg = cur_sum / float(cur_itn)
        
        return cur_avg, cur_sum, cur_itn

    
    def predict_all(self, data):
        """Returns the predicted values for some list of data points."""
        return map(lambda x: self.lookup(x), data)
        
    
    def find_weakest(self):
        """Finds the smallest value of alpha and 
        the first branch for which the full tree 
        does not minimize the error-complexity measure."""
        if (self.right == None):
            return -float("Inf"), [self]
        g_error, intnl_nodes = self.get_cost_params()
        alpha = g_error / float(intnl_nodes)
        alpha_right, tree_right = self.right.find_weakest()
        alpha_left, tree_left = self.left.find_weakest()
        smallest_alpha = max(alpha, alpha_right, alpha_left)
        smallest_trees = []
        # if there are multiple weakest links collapse all of them
        if smallest_alpha == alpha:
            smallest_trees.append(self)
        if smallest_alpha == alpha_right:
            smallest_trees = smallest_trees + tree_right
        if smallest_alpha == alpha_left:
            smallest_trees = smallest_trees + tree_left
        return smallest_alpha, smallest_trees
    
    
    def prune_tree(self):
        """Finds {a1, ..., ak} and {T1, ..., Tk},
        the sequence of nested subtrees from which to 
        choose the right sized tree."""
        trees = [copy.deepcopy(self)]
        alphas = [0]
        new_tree = copy.deepcopy(self)
        while 1:
            alpha, nodes = new_tree.find_weakest()
            for node in nodes:
                node.right = None
                node.left = None
            trees.append(copy.deepcopy(new_tree))
            alphas.append(alpha)
            # root node reached 
            if (node.start == True):
                break
        return alphas, trees

    
    def get_cost_params(self):
        """Returns the branch error and number of nodes."""
        if self.right == None:
            return 0.0, 0
        right_error, right_num = self.right.get_cost_params()
        left_error, left_num = self.left.get_cost_params()
        error = self.error + right_error + left_error
        internal_nodes = right_num + left_num + 1
        return error, internal_nodes


    def get_length(self):
        """Returns the length of the tree."""
        if self.right == None:
            return 1
        right_len = self.right.get_length()
        left_len = self.left.get_length()
        return max(right_len, left_len) + 1

    def __drawTree(self, s_width, s_height, window, tree_length, horz_step, depth, d_nodes):
        """Draws an image of the tree using pygame."""
        branch = (s_width + horz_step, s_height)
        pygame.draw.line(window, BLACK, (s_width, s_height), branch)
        if self.right == None:
            p = tree_font.render("Number of Instances: %d" % self.num_points, 1, BLACK)
            p2 = tree_font.render("Accuracy: %.3f" % self.morta, 1, BLACK)
            window.blit(p, (s_width + horz_step - 10*ft_size, s_height))
            window.blit(p2, (s_width + horz_step - 10*ft_size, s_height + ft_size))
            return
        if self.split_lab != None:
            lab = tree_font.render("%s" % self.split_lab, 1, RED)
            lab2 = tree_font.render("Number of Instances: %d" % self.num_points, 1, RED)
            lab3 = tree_font.render("Accuracy: %.3f" % self.morta, 1, RED)
        else:
            lab = tree_font.render("%d" % self.split_var, 1, RED)
            lab2 = tree_font.render("Number of Instances: %d" % self.num_points, 1, RED)
            lab3 = tree_font.render("Accuracy: %.3f" % self.morta, 1, RED)
        window.blit(lab, (s_width + (horz_step / 2), s_height))
        window.blit(lab2, (s_width + (horz_step / 2), s_height+ft_size))
        window.blit(lab3, (s_width + (horz_step / 2), s_height+2*ft_size))
        vert_step = d_nodes * 2 ** (tree_length - depth - 2)
        right_start = (s_width + horz_step, s_height - vert_step)
        left_start = (s_width + horz_step, s_height + vert_step)
        r_val = tree_font.render(">%.5f" % self.split_val, 1, RED)
        r_val2 = tree_font.render("Contribution: %.5f" % self.right_wt, 1, RED)
        window.blit(r_val, right_start)
        window.blit(r_val2, (s_width + horz_step, s_height - vert_step + ft_size))
        l_val = tree_font.render("<%.5f" % self.split_val, 1, RED)
        l_val2 = tree_font.render("Contribution: %.5f" % self.left_wt, 1, RED)
        window.blit(l_val, left_start)
        window.blit(l_val2, (s_width + horz_step, s_height + vert_step + ft_size))
        pygame.draw.line(window, BLACK, branch, right_start)
        pygame.draw.line(window, BLACK, branch, left_start)
        self.right.__drawTree(right_start[0], 
            right_start[1], window, tree_length, horz_step, depth + 1, d_nodes)
        self.left.__drawTree(left_start[0], 
            left_start[1], window, tree_length, horz_step, depth + 1, d_nodes)
        return

    
    def display_tree(self, save = False, filename = "image.jpg", view=True, height=6000, width=18000):
        """Wrapper function to draw the tree.
            If save is set to True will save the image to filename.
            If view is set to True will display the image."""
        nodes = self.get_cost_params()[1]
        tree_length = self.get_length()
        d_nodes = ((height - 40.0) / (2**(tree_length - 1))) 
        horz_step = ((width - 80) / tree_length)
        depth = 1
        window = pygame.display.set_mode((width, height))
        window.fill(WHITE)
        self.__drawTree(40, height / 2.0, window, tree_length, horz_step, depth, d_nodes)
        if save:
            pygame.image.save(window, filename)
        while view:
            for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     view = False
            pygame.display.flip()
        pygame.display.quit()
        return

def cvt(df_in, dm_in, id_p, max_depth = 500, Nmin = 100):
    # Build a contribution tree and use held-out data to validate
    df_train_cv = []
    df_test_cv = []
    dm_train_cv = []
    dm_test_cv = []


    #20% data for held-out validation
    split_n = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    random.shuffle(id_p)
    id_cv = split_n(id_p, (len(id_p)/5)+1)

    
    df_train_cv.append(df_in[-df_in['pid_contrb'].isin(id_cv[0])])
    df_test_cv.append(df_in[df_in['pid_contrb'].isin(id_cv[0])])
    dm_train_cv.append(dm_in[-dm_in['pid_visit'].isin(id_cv[0])])
    dm_test_cv.append(dm_in[dm_in['pid_visit'].isin(id_cv[0])])


    t_vs = []
    alpha_vs = []


    df_cur = df_train_cv[0]
    dm_cur = dm_train_cv[0]
    full_tree_v = grow_ctb_tree(df_cur, dm_cur, 0, max_depth = max_depth, Nmin = Nmin, starting = True)
    alphas_v, trees_v = full_tree_v.prune_tree()
    t_vs.append(trees_v)
    alpha_vs.append(alphas_v)


    min_R = float("Inf")
    min_ind = 0

    trees = trees_v


    for i in range(len(trees)):
        R_k = 0
        for j in range(len(t_vs)):
            print 'Cross validating, Tree: ' + str(i+1) + '/' + str(len(trees)) + ', round: ' + str(j+1) + '/' + str(len(t_vs))
            cur_avg, cur_sum, cur_itn= trees[i].lookup_df(df_test_cv[j])
            R_k = R_k + cur_sum
        print 'G value: ' + str(R_k)
        if (R_k < min_R):
            min_R = R_k
            min_ind = i


    return trees, min_ind

    

def error_function(split_point, split_var, data):
    """Function to minimize when choosing split point."""
    data1 = []
    data2 = []
    for i in data:
        if i[split_var] <= split_point:
            data1.append(data[i])
        else:
            data2.append(data[i])
    return region_error(data1) + region_error(data2)  
    
def region_error(data):
    """Calculates sum of squared error for some node in the regression tree."""
    data = numpy.array(data)
    return numpy.sum((data - numpy.mean(data))**2)



def error_func(split_point, ctbs, vals, Nmin):
    """Function to minimize when choosing split point."""
    ctb1 = []
    ctb2 = []
    val1 = []
    val2 = []

    condition = vals <= split_point
    val1 = vals[condition]
    ctb1 = ctbs[condition]
    val2 = vals[~condition]
    ctb2 = ctbs[~condition]
    
    if (len(val1) < Nmin) or (len(val2) < Nmin):
        return 10
    else:
        
        err1 = sum(ctb1) / float(len(ctb1))
        err2 = sum(ctb2) / float(len(ctb2))

        return -abs(err1 - err2)


def grow_ctb_tree(df, dm, depth, max_depth = 500, Nmin = 5, starting = False):
    # Function to grow a contribution ttee
    start = time.time()
    print 'Depth: ' + str(depth)
    
    root = Tree(sum(dm['lbl_visit']/float(len(dm))), 0.0, starting, len(dm))
    root.morta = sum(dm['lbl_visit'])/float(len(dm))
    
    n_pat = len(df['pid_contrb'].unique())

    if n_pat <= Nmin:
        return root

    if depth >= max_depth:
        return root

    varss = df['var'].unique()

    num_vars = len(varss)

    cand_vars = range(num_vars)

    min_error = -1
    min_split = -1
    split_var = -1

    for i in cand_vars:

        cur_var = varss[i]

        cur_df = df[df['var'] == cur_var]

        cur_vals = numpy.array(cur_df['val'])
        cur_ctbs = numpy.array(cur_df['ctb_contrb'])
         

        if len(cur_df) == n_pat:
            continue

        cur_vals = numpy.pad(cur_vals, (0, n_pat - len(cur_vals)%n_pat), 'constant')

        cur_ctbs = numpy.pad(cur_ctbs, (0, n_pat - len(cur_ctbs)%n_pat), 'constant')
        

        var_space  = list(numpy.unique(cur_vals))
        
        split, error, ierr, numf = scipy.optimize.fminbound(error_func, min(var_space), max(var_space), args = (cur_ctbs, cur_vals, Nmin), full_output = 1) 

        if error == 10:
            continue
        
        elif ((error < min_error) or (min_error == -1)):
            min_error = error
            min_split = split
            split_var = i

    if split_var == -1:
        return root

    root.split_var = split_var

    root.split_val = min_split

    root.split_lab = varss[split_var]
    root.num_points = n_pat
    root.error = min_error

    dfs = df[df['var'] == varss[split_var]]

    id1 = list(dfs[dfs['val'] > min_split]['pid_contrb'])

    dfs1 = df[df['pid_contrb'].isin(id1)]

    dfs2 = df[-df['pid_contrb'].isin(id1)]
    
    dm1 = dm[dm['pid_visit'].isin(id1)]
    dm2 = dm[-dm['pid_visit'].isin(id1)]

    ctb1sum = dfs1[dfs1['var'] == varss[split_var]]['ctb_contrb'].mean()

    if len(numpy.array(dfs1[dfs1['var'] == varss[split_var]]['ctb_contrb'])) == 0:
        ctb1sum = 0.0

    ctb2sum = dfs2[dfs2['var'] == varss[split_var]]['ctb_contrb'].mean()

    if len(numpy.array(dfs2[dfs2['var'] == varss[split_var]]['ctb_contrb'])) == 0:
        ctb2sum = 0.0


    df_l = dfs2
    df_r = dfs1
    dm_l = dm2
    dm_r = dm1
    root.left_wt = ctb2sum
    root.right_wt = ctb1sum
    
    end = time.time()
    print varss[split_var] + ' selected, time elapsed: ' + str(end - start) + ', error: ' + str(root.error)

    root.left = grow_ctb_tree(df_l, dm_l, depth + 1, max_depth = max_depth, Nmin = Nmin)
    
    root.right = grow_ctb_tree(df_r, dm_r, depth + 1, max_depth = max_depth, Nmin = Nmin)
        
    return root