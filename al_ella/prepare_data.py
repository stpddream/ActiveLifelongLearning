from scipy.io import loadmat
from data_process import load_landset
from data_process import load_dat
from data_process import gen_init
import numpy as np
import pickle as pk
from config import T
from util import dat_size
import util

source = 'fera_forella'

## Load From Source ##
train_dat, test_dat = load_dat(source)

print "init dat"
init_dat, pool_dat = gen_init(train_dat, 5)

print "train size", dat_size(train_dat)
print "test  size", dat_size(test_dat)
print "pool  size", dat_size(pool_dat)
print "init  size", dat_size(init_dat)

print "----> Writing to files..."

# Save to files
pool_f = open("data/pool", "wb")
pk.dump(pool_dat, pool_f)
pool_f.close()

test_f = open("data/test", "wb")
pk.dump(test_dat, test_f)
test_f.close()

init_f = open("data/init", "wb")
pk.dump(init_dat, init_f)
init_f.close()

print "----> %s Data Preparation Done." % source
### Data Summary ####
# counts = np.zeros(2)

# print "----> Loading & Splitting data..."
# land_train, land_test = load_landset()

# print "init dat"
# init_dat, pool_dat = gen_init(land_train, 5)

# print "train size", dat_size(land_train)
# print "test  size", dat_size(land_test)
# print "pool  size", dat_size(pool_dat)
# print "init  size", dat_size(init_dat)

# print "----> Writing to files..."

# # Save to files
# pool_f = open("data/pool", "wb")
# pk.dump(pool_dat, pool_f)
# pool_f.close()

# test_f = open("data/test", "wb")
# pk.dump(land_test, test_f)
# test_f.close()

# init_f = open("data/init", "wb")
# pk.dump(init_dat, init_f)
# init_f.close()

# print "----> Landmine Data Preparation Done."

# # # ###### File Load Tests ######
# # # t_pool_f = open("data/pool", "rb")
# # t_pool_dat = pk.load(t_pool_f)
# # t_pool_f.close()
# # print "pool size", dat_size(t_pool_dat)

# # # t_init_f = open("data/init", "rb")
# # # t_init_dat = pk.load(t_init_f)
# # # t_init_f.close()
# # # print "init size", dat_size(t_init_dat)

# # t_test_f = open("data/test", "rb")
# t_test_dat = pk.load(t_test_f)
# t_test_f.close()
