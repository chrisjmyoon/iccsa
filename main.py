import argparse
import pdb
import experiments
import utils



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('conf_path', type=str, metavar='conf_path')
    args = parser.parse_args()
    K = 5
    for k in range(K):
        config = utils.configs.BaseConfig(args.conf_path)
        experiment = experiments.toy.ToyExperiment(config, k)
        experiment.run()
    pdb.set_trace()

if __name__ == '__main__':
    main()