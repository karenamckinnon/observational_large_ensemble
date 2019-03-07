#!/bin/bash/

datadir=/glade/work/mckinnon

# BEST
best_url=http://berkeleyearth.lbl.gov/auto/Global/Gridded/Complete_TAVG_LatLong1.nc
best_readme=http://berkeleyearth.lbl.gov/auto/Global/Gridded/Gridded_README.txt

mkdir -p "$datadir"/BEST
wget -N -P "$datadir"/BEST $best_url
wget -N -P "$datadir"/BEST $best_readme

# GPCC
gpcc_url=ftp://ftp.cdc.noaa.gov/Datasets/gpcc/full_v7/precip.mon.total.1x1.v7.nc

mkdir -p "$datadir"/GPCC
wget -N -P "$datadir"/GPCC $gpcc_url

# SLP from 20CRv2
20cr_url=ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2c/Monthlies/monolevel/prmsl.mon.mean.nc

mkdir -p "$datadir"/20CRv2c
wget -N -P "$datadir"/20CRv2c $20cr_url

# SST modes
enso_url=https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/nino34.long.data
pdo_url=http://research.jisao.washington.edu/pdo/PDO.latest.txt
amo_url=https://www.esrl.noaa.gov/psd/data/correlation/amon.us.long.data

mkdir -p "$datadir"/SSTmodes
wget -N -P "$datadir"/SSTmodes $enso_url
wget -N -P "$datadir"/SSTmodes $pdo_url
wget -N -P "$datadir"/SSTmodes $amo_url
