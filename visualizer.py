import pandas as pd
import matplotlib.pyplot as plt


class Visualizer():

    def __init__(self, *args, **kwargs):
        self.dataframes = {}
        self.index_column = kwargs.get('index_column', "epoch")
        filename = kwargs.get('file_name', None)
        if filename is not None:
            df = pd.DataFrame.from_csv(filename, index_col=self.index_column)
            self.dataframes = {col: df[[col]].reset_index().dropna() for col in df.columns}

    def add_log(self, epoch_num_or_run, data):
        """
        :param epoch_num_or_run:
        :data: tuple (name, value)
        """
        col_name = data[0]
        value = data[1]
        if not (col_name in self.dataframes.keys()):
            self.dataframes[col_name] = pd.DataFrame(data={self.index_column: [epoch_num_or_run], col_name: [value]})
        else:
            df = self.dataframes[col_name]

            if not df.loc[df[self.index_column] == epoch_num_or_run].empty:
                raise ValueError("Epoch {} for column {} already logged !".format(epoch_num_or_run, col_name))

            df = df.append(pd.DataFrame(data={self.index_column: [epoch_num_or_run], col_name: [value]}))
            self.dataframes[col_name] = df

    def plot(self):
        plt.figure(figsize=(18, 8))
        ax = plt.gca()

        columns = self._get_columns()
        df = self._get_dataframe()
        df = df.reset_index()
        for col in columns:
            df.plot(kind='line', x=self.index_column, y=col, ax=ax)

        plt.show()

    def save(self, file_name):
        df = self._get_dataframe()
        df.to_csv(file_name, index=self.index_column)

    def _get_dataframe(self):
        df_list = [df.set_index(self.index_column) for df in self.dataframes.values()]
        return pd.concat(df_list, axis=1)

    def _get_columns(self):
        return [col for col in self.dataframes.keys()]



