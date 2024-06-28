# Generated by Django 5.0.4 on 2024-06-19 07:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0017_rename_model_comb_id_modelwiseemployeeroute_model_comb_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelcombination',
            name='no_of_vehicles',
            field=models.IntegerField(default=10),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='modelcombination',
            name='first_level_algo',
            field=models.CharField(choices=[('pick', 'pick-up'), ('drop', 'drop_off')], max_length=255),
        ),
    ]
