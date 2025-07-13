from torch import nn


class CategorizedModule(nn.Module):
    @property
    def category(self):
        raise NotImplementedError()

    def check_category(self, category):
        if category is None:
            raise RuntimeError('Category is not specified in this checkpoint.\n'
                               'If this is a checkpoint in the old format, please consider '
                               'migrating it to the new format via the following command:\n'
                               'python scripts/migrate.py ckpt <INPUT_CKPT> <OUTPUT_CKPT>')
        elif category != self.category:
            raise RuntimeError('Category mismatches!\n'
                               f'This checkpoint is of the category \'{category}\', '
                               f'but a checkpoint of category \'{self.category}\' is required.')
