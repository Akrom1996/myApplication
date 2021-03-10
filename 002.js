'use strict';

const { FileSystemWallet, Gateway } = require('fabric-network');
const path = require('path');

const ccpPath = path.resolve(__dirname, '..', '..', 'first-network', 'connection-org1.json');

const channelName = "mychannel";
const smartContractName = "contract";

async function main(traderId, firstName, lastName, balance) {
    try {
        const appAdmin = traderId;
        // Create a new file system based wallet for managing identities.
        const walletPath = path.join('/home/akrom/fabric-samples/myBlockchain/application', 'wallet');
        const wallet = new FileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check to see if we've already enrolled the user.
        const userExists = await wallet.exists(appAdmin);
        if (!userExists) {
            console.log(`An identity for the user ${appAdmin} does not exist in the wallet`);
            console.log('Run the enrollAdmin.js application before retrying');
            return;
        }

        // Create a new gateway for connecting to our peer node.
        const gateway = new Gateway();
        await gateway.connect(ccpPath, { wallet, identity: appAdmin, discovery: { enabled: true, asLocalhost: true } });
        // Get the network (channel) our contract is deployed to.
        const network = await gateway.getNetwork(channelName);

        // Get the contract from the network.
        const contract = network.getContract(smartContractName);


        //Begin submitting transactions

        //Submit create a trader transaction
        console.log('\nSubmit AddTrader transaction.');
        const addTraderAResponse = await contract.submitTransaction('addTrader', traderId, firstName, lastName, balance);
        console.log('addTraderAResponse: ');
        console.log(addTraderAResponse.toString('utf8'));
        console.log('addTraderAResponse_JSON.parse: ');
        console.log(JSON.parse(addTraderAResponse.toString()));
        await gateway.disconnect();

    } catch (error) {
        console.error(`Failed to submit transaction: ${error}`);
        process.exit(1);
    }
}
main("nodeB", "Tom", "Willson", "0");
//module.exports.main = main;
