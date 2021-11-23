pipeline {
    agent { docker { image 'pmantini/assignment-cosc6380:latest' } }

    environment {
        PATH = "env/bin/:$PATH"
    }
    stages {
        stage('build') {
            steps {
                sh 'python dip_hw4_filter.py -i Lenna.png -f arithmetic_mean -n gaussian'
                sh 'python dip_hw4_filter.py -i Lenna.png -f arithmetic_mean -n gaussian -s 7'
                sh 'python dip_hw4_filter.py -i Lenna.png -f geometric_mean -n gaussian'
                sh 'python dip_hw4_filter.py -i Lenna.png -f geometric_mean -n gaussian -s 9'
                sh 'python dip_hw4_filter.py -i Lenna.png -f local_noise -n gaussian -s 9'
                sh 'python dip_hw4_filter.py -i Lenna.png -f median -n bipolar'
                sh 'python dip_hw4_filter.py -i Lenna.png -f median -n bipolar -npa 0.5 -npb 0.5'
                sh 'python dip_hw4_filter.py -i Lenna.png -f median -n bipolar -npa 0.5 -npb 0.5 -s 7'
                sh 'python dip_hw4_filter.py -i Lenna.png -f adaptive_median -n bipolar -npa 0.5 -npb 0.5'
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'output/**/*.* ', onlyIfSuccessful: true
        }
    }
}