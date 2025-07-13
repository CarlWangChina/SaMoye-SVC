import pathlib
from collections import OrderedDict

import click


@click.group()
def main():
    pass


@main.command(help='Migrate checkpoint files of MIDI-less acoustic models from old format')
@click.argument('input_ckpt', metavar='INPUT')
@click.argument('output_ckpt', metavar='OUTPUT')
@click.option('--overwrite', is_flag=True, show_default=True, help='Overwrite the existing file')
def ckpt(
        input_ckpt: str,
        output_ckpt: str,
        overwrite: bool = False
):
    input_ckpt = pathlib.Path(input_ckpt).resolve()
    output_ckpt = pathlib.Path(output_ckpt).resolve()
    assert input_ckpt.exists(), 'The input file does not exist.'
    assert overwrite or not output_ckpt.exists(), \
        'The output file already exists or is the same as the input file.\n' \
        'This is not recommended because migration scripts may not be stable, ' \
        'and you may be at risk of losing your model.\n' \
        'If you are sure to OVERWRITE the existing file, please re-run this script with the \'--overwrite\' argument.'

    import torch
    ckpt_loaded = torch.load(input_ckpt, map_location='cpu')
    if 'category' in ckpt_loaded:
        print('This checkpoint file is already in the new format.')
        exit(0)
    state_dict: OrderedDict = ckpt_loaded['state_dict']
    ckpt_loaded['optimizer_states'][0]['state'].clear()
    new_state_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('model.fs2'):
            # keep model.fs2.xxx
            new_state_dict[key] = state_dict[key]
        else:
            # model.xxx => model.diffusion.xxx
            path = key.split('.', maxsplit=1)[1]
            new_state_dict[f'model.diffusion.{path}'] = state_dict[key]
    ckpt_loaded['category'] = 'acoustic'
    ckpt_loaded['state_dict'] = new_state_dict
    torch.save(ckpt_loaded, output_ckpt)


@main.command(help='Migrate transcriptions.txt in old datasets to transcriptions.csv')
@click.argument('input_txt', metavar='INPUT')
def txt(
        input_txt: str
):
    input_txt = pathlib.Path(input_txt).resolve()
    assert input_txt.exists(), 'The input file does not exist.'
    with open(input_txt, 'r', encoding='utf8') as f:
        utterances = f.readlines()
    utterances = [u.split('|') for u in utterances]
    utterances = [
        {
            'name': u[0],
            'ph_seq': u[2],
            'ph_dur': u[5]
        }
        for u in utterances
    ]

    import csv
    with open(input_txt.with_suffix('.csv'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'ph_seq', 'ph_dur'])
        writer.writeheader()
        writer.writerows(utterances)


if __name__ == '__main__':
    main()
