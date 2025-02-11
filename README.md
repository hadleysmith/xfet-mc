# xfet-mc
Monte Carlo XFET simulations and CNR analysis of numerical mouse phantom and contrast-depth phantom.

This repository contains code for 1) generating data from XFET Monte Carlo (MC) simulations of both a numerical mouse phantom and contrast-depth phantom, and 2) analyzing CNR results.

Due to large file sizes, data is available upon request. Please email hadley.debrosse@osumc.edu or contact the La Riviere lab group.

"simulations" contains code needed to perform TOPAS XFET MC simulations. More information about how to acquire and use TOPAS here: https://www.topasmc.org/
  
  "moby" contains code to run XFET MC simulation of the numerical mouse phantom, MOBY. Within "moby", there is txt files to run the TOPAS simulations:
      
      "XFET_combined.txt" : parameter file.
      
      "poly120kVp_cutoff.txt" : truncated beam spectrum input file.
      
  The moby phantom was split into 13 disctinct materials, detailed in the parameter file. The original numerical mouse phantom is licensed and must be acquired. See more     information:      https://otc.duke.edu/technologies/4d-mouse-whole-body-moby-phantom-version-2-0/ 
  
  csv output files are generated by running "XFET_combined.txt". Once csv files are generated, file names are formatted as follows: "sideX" indicates the detector number of 6, and "Run_XXXX" indicates the beam position index.
 
  Due to comutational restrictions, the simulation was split into four smaller simulations. Thus "XFET_combined" must be run 4 times with 4 different seeds, and each simulation contains 1/4 of the total number of histories.

    "ddc-phantom" contains code to run XFET MC simulation of the contrast depth (DDC) phantom. 
       
        "depth_1-2" contains code for first two depths of the DDC phantom (3.25 mm and 28.75 mm).
       
        "depth_3-4" contains code for last two depths of DDC phantom (54.25 mm and 79.75 mm).
       
        "dose" contains code to score dose to phantom.
       
        "partial_FOV" contains code for XFET imaging the smaller FOV of the DDC phantom.

        within each of these four sub-folders, we have txt files to run the simulations:
          
          "XFET_combined.txt" : parameter file.
         
          "poly120kVp_cutoff.txt" : truncated beam spectrum input file.
          
          "poly120kVp.txt" : full beam spectrum input file.
         
          "DDC_gen4" : DDC phantom file.

      csv output files are generated by running "XFET_combined.txt". Once csv files are generated, file names are formatted as follows: "sideX" indicates the detector number of 6, and "Run_XXXX" indicates the beam position index. 5 different noise realizations can be run by changing the seed in "XFET_combined.txt".

"analysis/code" contains python scripts to analyze TOPAS MC data, once generated. File paths may need altering. CT data can be generated by using code found at:https://github.com/gjadick/xtomosim .

    "ReadCNRtopasdata_final.py" : main code for creating images and analyzing CNRs of DDC phantom.
    
    "ReadCNRtopasdata_zoom_in_phantom.py" : main code for creating images of partial FOV of DDC phantom.
    
    "ReadTOPAS_MOBY_Data_final.py" : main code for creating images and comparing CNRs of MOBY phantom.
    
    "plotSlices.py" :   Function for visualizing 3D slices of object.
    
    "get_CNR.py" : Function for extracting CNRs of phantoms in this study.
    
    "create_contrast_phantom.py" : Function to make voxelized DDC phantom (replicating TOPAS object) to use in CT simulations.
    
    "convert_coordinates.py" : function for converting cartesian to polar coordinates, and similar functions.

    
        
      
