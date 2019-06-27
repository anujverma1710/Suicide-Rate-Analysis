# Suicide-Rate-Analysis

A client-server system: used python for processing (server), D3 and Javascript/Jquery for visualisation (client)

data clustering and decimation 
<ul>
  <li>implemented random sampling and stratified sampling</li>
  <li>k-means clustering (optimized k using elbow method)</li>
 </ul>
dimension reduction
<ul>
  <li>found the intrinsic dimensionality of the data using PCA</li>
  <li>produced scree plot visualization and marked the intrinsic dimensionality</li>
  <li>obtained the three attributes with highest PCA loadings</li>
 </ul>
visualization (using dimension reduced data) 
<ul>
  <li>visualized data projected into the top two PCA vectors via 2D scatterplot</li>
  <li>visualized data via MDS (Euclidian & correlation distance) in 2D scatterplots</li>
  <li>visualized scatterplot matrix of the three highest PCA loaded attributes</li>
</ul>
