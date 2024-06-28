# Generated by Django 5.0.4 on 2024-06-18 08:20

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0013_alter_vehicle_path_data'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelCombination',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type_id', models.CharField(max_length=20)),
                ('first_level_algo', models.CharField(max_length=255)),
                ('second_level_algo', models.CharField(max_length=255)),
                ('analysis', models.TextField(blank=True, null=True)),
                ('shift', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.shift')),
            ],
        ),
        migrations.CreateModel(
            name='ModelWiseEmployeeRoute',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sequence_no', models.IntegerField()),
                ('employee', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.employee')),
                ('model_comb_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.modelcombination')),
                ('vehicle', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='users.vehicle')),
            ],
        ),
    ]
