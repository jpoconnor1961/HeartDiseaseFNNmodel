# HeartDiseaseFNNmodel

Download & Install Instructions for the submitted Heart Disease FNN Model Project Folder/Files in R:

- In your web browser enter the following URL into the address bar: https://github.com/jpoconnor1961/HeartDiseaseFNNmodel
- In the github HeartDiseaseFNNmodel repository, click on the green button that says "Code", and then click on the "Download ZIP" choice in the dropdown menu.
- In your FileManager, extract the zip folder "HeartDiseaseFNNmodel-main" that was downloaded. This will create a new "HeartDiseaseFNNmodel-main" folder that will contain another nested "HeartDiseaseFNNmodel-main" folder that will contain a "harvard_oconnor_FNNmodel" folder which contains all of the project files.
- In your FileManager, Copy or Move the "harvard_oconnor_FNNmodel" folder (which contains all of the project files) to the Project folder for your R application.
- In your R application, select "New Project" and then select an "Existing Directory". Next, browse/navigate to the "harvard_oconnor_FNNmodel" folder to select that as the working directory, and then select the "Create Project" button at the bottom-right-side of the wizard dialog box.
- The "harvard_oconnor_FNNmodel" working directory folder in your R application will contain the following:
   - harvard_oconnor_FNNmodel_PDF file
   - harvard_oconnor_FNNmodel_Rcode file
   - train.csv
   - test.csv
   - harvard_oconnor_FNNmodel_RMD markdown file
   - 13 PNG image files to support the figures coded in the markdown file
   - 1 JPG image file to support a figure coded in the markdown file
   - ref.bibtex file that supports the markdown file
- If you have problems with "Knit to PDF" from the open markdown file tab, you may need to install or update your LaTeX package. In your R application, install the R package "tinytex" to make sure it is up to date (this is a LaTeX management package, not LaTeX itself). Open the tinytex package in R and run the install_tinytex() function (with the default settings) which installs LaTeX itself from a repository. After the LaTeX install, re-try to Knit to PDF again and the (still open) tinytex R package should kickin during the knitting process and help R to render the PDF by automatically installing any missing LaTeX packages or files. Depending on the scope of the original LaTeX problem, it may take a few re-trys of Knit to PDF for the tinytex manager to find and correct all of the missing packages, files, paths, etc.
