import pandas as pd


def main():
    probs = {}
    probs['se_resnext50_32x4d'] = pd.read_csv('probs/se_resnext50_32x4d_080922.csv')['diagnosis'].values
    probs['se_resnext101_32x4d'] = pd.read_csv('probs/se_resnext101_32x4d_081208.csv')['diagnosis'].values
    probs['senet154'] = pd.read_csv('probs/senet154_082510.csv')['diagnosis'].values

    test_df = pd.read_csv('inputs/test.csv')
    test_df['diagnosis'] = 0.4 * probs['se_resnext50_32x4d'] + 0.3 * probs['se_resnext101_32x4d'] + 0.3 * probs['senet154']
    test_df.to_csv('probs/weighted_average.csv', index=False)


if __name__ == '__main__':
    main()
