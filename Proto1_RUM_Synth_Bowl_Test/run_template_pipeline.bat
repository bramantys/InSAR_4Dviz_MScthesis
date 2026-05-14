@echo off
setlocal

echo ============================================================
echo RUM 4D TEMPLATE PIPELINE
echo ============================================================
echo.

echo Step 01 - Convert source to WGS84 GeoJSON
python pipeline_template\01_convert_source_to_geojson.py || goto fail

echo Step 02 - Prepare vertical epochs from velocity
python pipeline_template\02_prepare_vertical_epochs_from_velocity.py || goto fail

echo Step 03 - Build corrected RUM footprints
python pipeline_template\03_build_footprints.py || goto fail

echo Step 04 - Enhance vertical sigma optional
python pipeline_template\04_enhance_vertical_sigma_optional.py || goto fail

echo Step 05 - Validate prepared inputs
python pipeline_template\05_validate_prepared_inputs.py || goto fail

echo Step 06 - Pack vertical series
python pipeline_template\06_pack_vertical_series.py || goto fail

echo Step 08 - Build tile index
python pipeline_template\08_build_tile_index.py || goto fail

echo Step 09 - Export epoch axis
python pipeline_template\09_export_epoch_axis.py || goto fail

echo Step 10 - Build blank cells
python pipeline_template\10_build_blank_cells.py || goto fail

echo Step 11 - Build height texture
python pipeline_template\11_build_height_texture.py || goto fail

echo Step 12 - Build real RUM caps B3DM
python pipeline_template\12_build_real_caps_b3dm.py || goto fail

echo Step 13 - Build blank caps B3DM
REM python pipeline_template\13_build_blank_caps_b3dm.py || goto fail

echo Step 14 - Build walls B3DM
python pipeline_template\14_build_walls_b3dm.py || goto fail

echo Step 16 - Build horizontal field
python pipeline_template\16_build_horizontal_field.py || goto fail

echo Step 17 - Check horizontal field
python pipeline_template\17_check_horizontal_field.py || goto fail

echo Step 18 - Check horizontal uncertainty
python pipeline_template\18_check_horizontal_uncertainty.py || goto fail

echo.
echo ============================================================
echo PIPELINE COMPLETE
echo ============================================================
echo.
goto end

:fail
echo.
echo ============================================================
echo PIPELINE FAILED
echo ============================================================
echo Check the error above.
exit /b 1

:end
endlocal