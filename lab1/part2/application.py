import pickle

import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import tensorflow as tf

import implementation as imp


def draw_histogram(weights_file):
    file = open(weights_file, 'rb')
    weights = pickle.load(file)

    a = weights[0][0].flatten()
    b = weights[2][0].flatten()
    c = weights[5][0].flatten()
    plt.hist(x=a, bins=np.linspace(-2, 2, 21), color='blue',
             alpha=1.0, rwidth=0.85)
    plt.hist(x=b, bins=np.linspace(-2, 2, 21), color='blue',
             alpha=0.5, rwidth=0.85)
    plt.hist(x=c, bins=np.linspace(-2, 2, 21), color='blue',
             alpha=0.3, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Quantity')
    plt.ylabel('Weight')
    plt.xticks(np.linspace(-2, 2, 11))
    plt.ylim(ymax=30)
    plt.savefig('histogram.png')
    plt.show()


def draw_boxplot():
    file = open('lab1/part2/node_num_noise009', 'rb')
    results = pickle.load(file)
    log_results = np.log(results).swapaxes(0, 1)
    plt.figure(figsize=(8, 4))
    show_outliers = True
    plt.boxplot(log_results, showfliers=show_outliers)
    plt.xlabel("Number of nodes")
    plt.ylabel("log MSE")
    plt.savefig('boxplot.png')
    plt.show()


def plot_prediction(true, pred, logdir, epoch):
    plt.figure()
    plt.plot(pred, color='blue')
    plt.plot(true, color='red')
    plt.savefig(logdir + str(int(epoch)) + '.png')
    # plt.show()


def plot_data_with_preditction(X, pred, logdir, epoch):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    plt.plot(X, color='black', linewidth=0.7)
    x_val = np.linspace(1300, 1500, 200)
    plt.plot(x_val, pred, color='red', linewidth=0.7)
    plt.xticks([0, 300, 1100, 1300, 1500])
    plt.yticks(np.linspace(0, 1.5, 4))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(logdir + str(int(epoch)) + '.png', dpi=300)
    # plt.show()


def plot_error(true, pred, logdir, epoch):
    plt.figure()
    res = np.abs(np.subtract(true, pred.squeeze()))
    plt.plot(res, color='red')
    plt.savefig(logdir + "err_" + str(int(epoch)) + '.png')


if __name__ == '__main__':

    # Setup grid search
    params = {'layer_dim': [[5, 7, 1], [5, 7, 5, 1]],
              'lr': [0.1],
              'regularization': [0],
              'noise': [0]}
    configs = imp.product_dict(params)
    configs, configs_ = it.tee(configs)

    iterations = 1
    weights_matrix = []
    results = []

    # Iterate over all configurations
    for conf in configs:
        result_node = []
        for i in range(iterations):
            noise = conf.get('noise')
            x_train, y_train, x_val, y_val, x_test, y_test, X = imp.generate_data(add_noise=False, sigma=noise, plot=False)

            lr = conf.get("lr")
            reg = conf.get("regularization")
            layer_dim = conf.get("layer_dim")
            model = imp.create_model(layer_dim, reg)

            conf_str = '_'.join([str(item) for item in layer_dim ])
            log_dir = "lab1/part2/logs/"
            log_name = f"nodes{conf_str}_reg{reg}_lr{lr}_run{i}/"
            log_dir = log_dir + log_name
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

            # callback for validation visualization
            def testmodel(epoch, logs):
                if epoch % 100 == 0:
                    pred = model.predict(x_test, batch_size=200)
                    plot_data_with_preditction(X, pred, log_dir, epoch / 100)
                    plot_error(y_test, pred, log_dir, epoch/100)

            testmodelcb = tf.keras.callbacks.LambdaCallback(on_epoch_end=testmodel)
            earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                                  patience=1000)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
            model.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['mae', 'mse'])

            batch_size = len(x_train)
            history = model.fit(x=x_train,
                                y=y_train,
                                epochs=30000,
                                batch_size=batch_size,
                                validation_data=(x_val, y_val),
                                callbacks=[tensorboard_callback, testmodelcb])
            pred = model.predict(x_test)

            result_node.append(np.min(history.history["loss"]))
        results.append(result_node)
    file = open('logs/results', 'wb')
    pickle.dump(results, file)
    file.close()
