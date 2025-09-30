@description('Name of the App Services NSG')
param nsgName string = 'nsg-app-services'

@description('Location for the NSG')
param location string = resourceGroup().location

@description('Account coding for billing and tracking')
param accountCoding string = 'EPIC-001'

@description('Billing group for cost allocation')
param billingGroup string = 'EPIC-Team'

@description('Ministry or department name')
param ministryName string = 'Citizens Services'

@description('Additional custom tags to merge with standard tags')
param customTags object = {}

// Import the shared tags function
import { generateTags } from '../../shared/tags.bicep'

// Generate tags using the shared function
var tags = generateTags(accountCoding, billingGroup, ministryName, union(customTags, {
  Component: 'NetworkSecurity'
  Purpose: 'AppServices'
}))

// Create App Services NSG
resource appServicesNSG 'Microsoft.Network/networkSecurityGroups@2024-07-01' = {
  name: nsgName
  location: location
  tags: tags
  properties: {
    securityRules: [
      // Default rules will be added here based on your templates
    ]
  }
}

// Outputs
output nsgId string = appServicesNSG.id
output nsgName string = appServicesNSG.name
output nsgResourceId string = appServicesNSG.id
