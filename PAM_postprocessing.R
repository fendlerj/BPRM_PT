library(cluster)

#Path to find the similarity matrix in a csv format
path_sim <- "path"
#Path to find the data in a csv format
path_csv <- "path"
#Path where to write the results
path_csv <- path

#Load data and similarity matrix
sim_mat <- read.csv(path_sim, header = FALSE)
data <- read.csv(patg_csv, row.names = 1)


#Maximum number of clusters tested 
Kmax <- 20

#Compute the silhouette of the clustering from a partitioning around medoids (PAM)
# procedure for a number of clusters between 2 and Kmax
sil <- rep(0, Kmax)
for(k in c(2:Kmax)){
  print(k)
  pam_result <- pam(1 - as.matrix(sim_mat), k = k, diss = TRUE)
  sil[k] <- pam_result$silinfo$avg.width
}

#Plot the value of the silhoutte in function of the number of clusters
plot(sil)

#Select the optimal number of clusters (i.e., the number of clusters for which the 
# silhouttte is maximum)
k_opt <- which(max(sil) == sil)

#Compute the clustering for the optimal number of clusters
cluster_pam <- pam(1 - as.matrix(sim_mat), k = k_opt, diss = TRUE)
summary(as.factor(cluster_pam$clustering))
write.csv(cluster_pam$clustering, paste0(path_results, "/clustering.csv"))


#Compare the obtained clustering with the truth (only for with simulated data)
# Note : the clusters labels might not match between the truth and the estimated clustering
table(cluster_pam$clustering, data$C)


# Note : the PAM procedure is determinist, this code contains no randomness!