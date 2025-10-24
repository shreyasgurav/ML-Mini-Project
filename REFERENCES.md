# References and Bibliography

## Primary Research Paper
**Title**: Crop Yield Prediction Using Hybrid Machine Learning Approach: A Case Study of Lentil (Lens culinaris Medik.)  
**Authors**: Pankaj Das, Girish Kumar Jha*, Achal Lama, Rajender Parsad  
**Affiliations**: 
- ICAR-Indian Agricultural Statistics Research Institute, New Delhi 110012, India
- ICAR-Indian Agricultural Research Institute, New Delhi 110012, India  
**Journal**: Agriculture  
**Volume**: 13(3)  
**Pages**: 596  
**Year**: 2023  
**DOI**: https://doi.org/10.3390/agriculture13030596  
**Submission**: 16 September 2022 / Revised: 21 November 2022 / Accepted: 26 November 2022 / Published: 28 February 2023  

## Methodology Substitution Justification

### Academic Rationale for Component Replacements

This implementation adapts the original MARS-ANN and MARS-SVR methodology using standard Scikit-learn components due to installation constraints while maintaining the core hybrid approach:

#### 1. Feature Selection Proxy
**Original Component**: MARS Feature Selection (pyearth library)  
**Replacement Component**: Polynomial Features + SelectKBest (F-test)  
**Rationale**: Replaces MARS's implicit variable selection with a standard, explicit statistical selection method (F-test for regression) available in Scikit-learn (f_regression). This maintains the feature selection capability while using established statistical methods.

#### 2. Model Substitution  
**Original Component**: ANN Model (TensorFlow/Keras)  
**Replacement Component**: Random Forest Regressor  
**Rationale**: Replaces the complex non-linear deep learning model with a robust non-linear ensemble tree model (Random Forest) available in Scikit-learn (RandomForestRegressor). This substitution maintains non-linear modeling capability while providing comparable performance and interpretability.

#### 3. Academic Justification
Both substitution techniques are standard regression and feature selection tools provided by the Scikit-learn Python library, ensuring:
- **Methodological Consistency**: Maintains hybrid approach (feature selection + ML models)
- **Statistical Rigor**: Uses established statistical methods (F-test, ensemble learning)
- **Reproducibility**: Standard library components ensure consistent results
- **Academic Acceptability**: Well-documented methods in machine learning literature

## Technical References

### Machine Learning Algorithms
1. **Support Vector Regression (SVR)**
   - Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag.
   - Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14(3), 199-222.

2. **Random Forest**
   - Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

3. **Multivariate Adaptive Regression Splines (MARS)**
   - Friedman, J. H. (1991). Multivariate adaptive regression splines. *The Annals of Statistics*, 19(1), 1-67.
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (Chapter 9). Springer.

### Feature Selection Methods
4. **Statistical Feature Selection**
   - Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182.
   - Saeys, Y., Inza, I., & Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics*, 23(19), 2507-2517.

### Agricultural Data Analysis
5. **Crop Yield Prediction**
   - Pantazi, X. E., Moshou, D., & Alexandridis, T. (2016). Wheat yield prediction using machine learning and advanced sensing techniques. *Computers and Electronics in Agriculture*, 121, 57-65.
   - Jeong, J. H., Resop, J. P., Mueller, N. D., Fleisher, D. H., Yun, K., Butler, E. E., ... & Kim, S. H. (2016). Random forests for global and regional crop yield predictions. *PLoS One*, 11(6), e0156571.

6. **Lentil Crop Studies**
   - Kumar, J., Basu, P. S., Srivastava, E., Singh, S., Kumar, S., Basu, P., ... & Kumar, A. (2012). Phenotyping of traits imparting drought tolerance in lentil. *Crop and Pasture Science*, 63(6), 547-560.
   - Singh, M., Singh, B., & Singh, S. (2013). Genetic variability and correlation studies in lentil (*Lens culinaris* Medik.). *Legume Research*, 36(2), 157-161.

### Python Libraries and Tools
7. **Scikit-learn**
   - Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

8. **Pandas**
   - McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 445, 51-56.

9. **NumPy**
   - Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.

## Evaluation Metrics References
10. **Model Evaluation**
    - Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. *Climate Research*, 30(1), 79-82.
    - Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7(3), 1247-1250.

## Data Preprocessing References
11. **Feature Scaling**
    - Hsu, C. W., Chang, C. C., & Lin, C. J. (2003). A practical guide to support vector classification. *Technical Report*, Department of Computer Science, National Taiwan University.

12. **Cross-Validation and Train-Test Split**
    - Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence*, 2, 1137-1143.

## Additional Resources

### Online Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

### Machine Learning Courses
- Andrew Ng's Machine Learning Course (Coursera)
- Elements of Statistical Learning (Free Online Book)
- Python for Data Science (Various Platforms)

### Agricultural Data Sources
- [FAO Statistical Database](http://www.fao.org/faostat/)
- [USDA Crop Production Data](https://www.nass.usda.gov/)
- [ICARDA Lentil Database](https://www.icarda.org/)

## Citation Format

### For the Original Research Paper
```
Das, P.; Jha, G.K.; Lama, A.; Parsad, R. Crop Yield Prediction Using Hybrid Machine Learning Approach: A Case Study of Lentil (Lens culinaris Medik.). Agriculture 2023, 13, 596. https://doi.org/10.3390/agriculture13030596
```

### For This Implementation
```
[Your Name]. (2024). Crop Yield Prediction using Hybrid MARS-Like Machine Learning Approach: 
Adaptation of Das et al. (2023) Methodology. ML Mini Project, [Institution Name].
```

## License and Usage
This project is created for educational purposes. Please cite appropriately if used in academic work or research.
