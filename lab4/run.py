from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    # added vars for 4.1a and 4.1b
    n_hidden = [500]

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    rbm_errors = []

    for hidden in n_hidden:

        print('RBM with {} hidden units:\n'.format(hidden))
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                         ndim_hidden=hidden,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=20
        )

        epoch_errors = rbm.cd1(visible_trainset=train_imgs, n_iterations=10)
        rbm_errors.append(epoch_errors)
    # 7x100
    print(np.shape(rbm_errors))
    # print(rbm_errors[0])

    plt.figure()
    plt.title('Comparison of different RBM architectures performance')
    plt.ylabel('Mean reconstruction error')
    plt.xlabel('Training samples')
    for i,n in enumerate(n_hidden):
        plt.plot(np.arange(1, 600001, 6000), rbm_errors[i], label='N_hidden={}'.format(n))
    plt.legend()
    plt.show()
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
