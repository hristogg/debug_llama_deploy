Attempting to recreate llamadeploy bug
- requirements used is the same as used in core project
- instead of VectorSearch managed by Google using two simple retrievers ingesting paul_graham_essay split into two
The workflow does not make a lot of practical sense, but according to my understanding it should be capable to work fine :)

.env file should be present with PHOENIX_CLIENT_HEADERS and PHOENIX_COLLECTOR_ENDPOINT 
Deployment:

1) gcloud builds submit --region=europe-west3 --tag gcr.io/<project-id>>/llamadebug

2) gcloud run deploy llamadebugcont --image gcr.io/<project-id>>/llamadebug --memory 4Gi --cpu 2

Deploying the workflow to the cloudrun instance

1) llamactl --server <cloudruninstsance> status 

Expected result: 
    LlamaDeploy is up and running.
    Currently there are no active deployments

2) llamactl --server <cloudruninstance> deploy workflow_config.yml

Expected result:
    Deployment successful: testwork
    
3) llamactl --server <cloudruninstance> status

Expected result:
    LlamaDeploy is up and running.

    Active deployments:
    - testwork
4) python -m test_deployed

Expected results - when on local deployment the test_deployed works fine, however when put on Cloudrun - fails.




